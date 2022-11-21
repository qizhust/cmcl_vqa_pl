import torch
import json, os
from transformers import BeitModel, BeitConfig, BeitForMaskedImageModeling


def build_beit_model(original_config_dir, image_size):
    config = json.load(open(os.path.join(original_config_dir, 'config.json'), 'r'))
    if config['image_size'] == image_size:
        return BeitForMaskedImageModeling.from_pretrained(original_config_dir)
    else:
        beit_config = BeitConfig()
        config['image_size'] = image_size
        beit_config.update(config)
        return BeitForMaskedImageModeling(beit_config)


def beit_forward(beit_model, pixel_values):
    outputs = beit_model.beit(pixel_values)
    sequence_output = outputs[0]
    sequence_output = beit_model.layernorm(sequence_output)
    return sequence_output

def beit_adapt_position_encoding(model, patch_size, after):
    pos_key = 'vit_model.embeddings.position_embeddings'
    if model.get(pos_key, None) is None:
        return model
    else:
        assert len(model[pos_key].shape) == 3 and model[pos_key].shape[0]==1, \
            'original position embeddings supposed to be in shape [1, cnum_patches+1, emb_dim]'
        old_num_patches = model[pos_key].shape[-2] - 1
        new_num_patches = (after//patch_size)**2
        grid_before = int(old_num_patches**0.5)
        grid_after = int(new_num_patches**0.5)
        old_pos_emb = model[pos_key].squeeze(0)[1:].view(grid_before, grid_before, -1)
        new_pos_emb = torch.nn.functional.interpolate(old_pos_emb.permute((2, 0, 1)), size=(grid_after, grid_after), mode='bicubic')
        model[pos_key][0, 1:] = new_pos_emb.permute((1, 2, 0)).view(new_num_patches, -1)
        return model
