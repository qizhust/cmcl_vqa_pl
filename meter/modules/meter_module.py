import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

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

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"]*2)
            self.itm_score.apply(objectives.init_weights)

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
            elif 'detr' in config['vit']:  # use detr
                pass
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)


        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        meter_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if 'ViT' in config['vit']:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
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
        if self.training:
            img_list = ['image', 'false_image']
        else:
            img_list = ['image']

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

        if self.training and self.hparams.config['nce']:
            image_base = torch.cat([cls_feats_image_base['image'], cls_feats_image_base['false_image']], 0)
            neg_mask = build_connected_component(image_base, self.hparams.config.get('sample_k', 1))  # False for selected negative sample

            # get cls text embedding before multi-modal fusing
            cls_feats_text_base = self.cross_modal_text_pooler(text_embeds)

            # get qa encoding
            ans_labels = []
            for ans in batch['vqa_labels']:
                ans_labels.append(ans[0])  # use the first answer as gt, or the answer with the highest score
            ans_labels = torch.LongTensor(ans_labels).to(self.device)
            ans_emb = self.ans_embed(ans_labels)
            qa_encoded = self.qa_encoder(torch.cat([cls_feats_text_base, ans_emb], -1))

            # calculate InfoNCE loss
            for ele in ['image', 'false_image']:
                f_xc[ele] = torch.matmul(cls_feats_image_base[ele], qa_encoded.transpose(0, 1)).exp()

            f_xc_all = torch.cat([f_xc['image'], f_xc['false_image']], 0)  # col for c, row for im
            if not self.hparams.config['nce_multipos']:
                eye_mask = torch.cat([torch.eye(input_shape[0]), torch.zeros(input_shape[0], input_shape[0])], 1).to(device)
                sample_mask = ~neg_mask[:input_shape[0]] + eye_mask  # the negative samples outside the graph and the positive sample
                f_xc_all = torch.diagonal(f_xc['image'])/(f_xc_all * sample_mask.T).sum(0)
                loss_infonce = -(f_xc_all.log() * (1.0 - torch.tensor(batch['yes_type']).to(device))).sum()
            else:
                loss_infonce = -(f_xc_all / f_xc_all.sum(0)).log() * neg_mask[:input_shape[0]].T
                loss_infonce = loss_infonce.sum(0)/neg_mask[:input_shape[0]].sum(1)
                loss_infonce = (loss_infonce * (1.0 - torch.tensor(batch['yes_type']).to(device))).sum()
        else:
            loss_infonce = 0.0

        # cross-modal encoding
        x, y = text_embeds, image_embeds['image']
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks['image'])
            y1 = image_layer(y, x, extend_image_masks['image'], extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if 'ViT' in self.hparams.config['vit']:  # use clip-vit
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:  # use detr or swin
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "loss_infonce": loss_infonce
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Oracle Masked Language Modeling
        if "mlm_oracle" in self.current_tasks:
            ret.update(objectives.compute_mlm_oracle(self, batch))

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v*self.current_tasks_w[k] for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, self.hparams.config['log_dir'])
        meter_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return meter_utils.set_schedule(self)
