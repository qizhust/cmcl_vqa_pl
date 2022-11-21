import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import clip

from transformers.models.bert.modeling_bert import BertConfig, BertModel, BertForMaskedLM
from .bert_model import BertCrossLayer
from . import swin_transformer as swin
from . import heads, objectives, meter_utils
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@torch.no_grad()
def build_connected_component(image_base, k=1):
    image_sim = torch.matmul(image_base, image_base.transpose(0,1))
    image_norm = (image_base*image_base).sum(-1).sqrt().unsqueeze(-1)
    image_norm = torch.matmul(image_norm, image_norm.transpose(0,1))
    dist = image_sim/image_norm  # here dist means normalized similarity

    device = dist.device
    b = dist.size(0)
    dist = dist - torch.eye(b, b, device=device) * 2
    x = torch.arange(b, device=device).unsqueeze(1).repeat(1,1).flatten()
    y = torch.topk(dist, k, dim=1, sorted=False)[1]
    rx, ry = [], []
    for i in range(k):
        rxi = torch.cat([x, y[:, i]])
        ryi = torch.cat([y[:, i], x])
        rx.append(rxi)
        ry.append(ryi)
    rx = torch.cat(rx, 0).cpu().numpy()
    ry = torch.cat(ry, 0).cpu().numpy()
    v = np.ones(rx.shape[0])
    graph = csr_matrix((v, (rx, ry)), shape=(b,b))
    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    labels = torch.tensor(labels, device=device)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
    return mask


class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after=config['image_size']

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if 'ViT' in config['vit']:  # use clip-vit
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(
                        pretrained=True, config=self.hparams.config,
                    )

                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                    RobertaForMaskedLM.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])
                    BertForMaskedLM.from_pretrained(config['tokenizer'])

            torch.distributed.barrier()

        if 'ViT' in config['vit']:  # use clip-vit
            self.vit_model = build_model(config['vit'], resolution_after=resolution_after)
        else:  # use swin
            self.vit_model = getattr(swin, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        if 'roberta' in config['tokenizer']:
            self.text_transformer = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.text_transformer = BertModel.from_pretrained(config['tokenizer'])

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

            self.ans_embed = nn.Embedding(vs, config['qa_emb_dim'])
            self.ans_embed.apply(objectives.init_weights)
            self.qa_encoder = nn.Linear(hs+config['qa_emb_dim'], hs)
            self.qa_encoder.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if 'ViT' in config['vit']:  # use clip-vit
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)

        meter_utils.set_metrics(self)
        self.current_tasks = list()
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device='cuda:0')

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if 'ViT' in config['vit']:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        clip_model=None,
        clip_preprocess=None
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        # get text embeds from nlp module and cross_modal_text module
        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        image_embeds, image_masks, extend_image_masks, cls_feats_image_base, f_xc = {}, {}, {}, {}, {}
        # get image embeds from vision module and cross_modal_image module
        img_list = ['image', 'false_image']

        for ele in img_list:
            image_embeds[ele] = self.vit_model(batch[ele][0])
            image_embeds[ele] = self.cross_modal_image_transform(image_embeds[ele])
            image_masks[ele] = torch.ones((image_embeds[ele].size(0), image_embeds[ele].size(1)), dtype=torch.long, device=device)
            extend_image_masks[ele] = self.text_transformer.get_extended_attention_mask(image_masks[ele], image_masks[ele].size(), device)

            image_embeds[ele] = image_embeds[ele] + self.token_type_embeddings(torch.full_like(image_masks[ele], image_token_type_idx))

            # get cls image embedding before multi-modal fusing
            if 'ViT' in self.hparams.config['vit']:  # use clip-vit
                cls_feats_image_base[ele] = self.cross_modal_image_pooler(image_embeds[ele])
            else:  # use detr or swin
                avg_image_feats = self.avgpool(image_embeds[ele].transpose(1, 2)).view(image_embeds[ele].size(0), 1, -1)
                cls_feats_image_base[ele] = self.cross_modal_image_pooler(avg_image_feats)

        image_base = torch.cat([cls_feats_image_base['image'], cls_feats_image_base['false_image']], 0)
        neg_mask = build_connected_component(image_base)  # False for selected negative sample

        text_in = []
        for ques, ans, ans_score in zip(batch['text'], batch['vqa_answer'], batch['vqa_scores']):
            text_in.append(' '.join([ques, ans[np.argsort(ans_score)[-1]]]))
        text_in = clip.tokenize(text_in).to(device)

        image_tensor = []        
        for raw_im in batch['raw_Image']:
            image_tensor.append(self.clip_preprocess(raw_im).unsqueeze(0).to(device))
        for raw_im in batch['raw_false_Image']:
            image_tensor.append(self.clip_preprocess(raw_im).unsqueeze(0).to(device))
        image_tensor = torch.cat(image_tensor, 0)

        logits_per_image, logits_per_text = self.clip_model(image_tensor, text_in)
        probs = logits_per_image.softmax(dim=-1)

        ref_score = torch.diag(probs).mean().cpu().numpy().item()
        pos_score, neg_score = [], []
        for idx in range(input_shape[0]):
            pos_score.append(probs[neg_mask[idx]][:, idx].mean().cpu().numpy())
            neg_score.append(probs[~neg_mask[idx]][:, idx].mean().cpu().numpy())
        pos_score = sum(pos_score)/len(pos_score)
        neg_score = sum(neg_score)/len(neg_score)

        ret = {'ref_score': ref_score, 'pos_score': pos_score, 'neg_score': neg_score}

        return ret

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outs):
        pass

    def test_step(self, batch, batch_idx):
        output = self(batch)
        return output

    def test_epoch_end(self, outs):
        model_name = 'meter'  # ['baseline', 'ours_b', 'meter', 'ours_m']

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, self.hparams.config['log_dir'])

    def configure_optimizers(self):
        return meter_utils.set_schedule(self)
