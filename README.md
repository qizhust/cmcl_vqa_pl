# Cross-Modal Contrastive Learning for Robust Reasoning in VQA

This repo is an implementation upon [METER](https://github.com/zdou0830/METER) backbone with PyTorch Lightning. [Here](https://github.com/qizhust/cmcl_vqa) is an implementation in PyTorch.

## Data preparation and pretrained models

Please follow [METER](https://github.com/zdou0830/METER) and [ViLT](https://github.com/dandelin/ViLT/blob/master/DATA.md) to prepare the datasets and download the pretrained checkpoints released by [METER](https://github.com/zdou0830/METER). Modify ```data_root``` and ```log_dir``` in ```config.py```.

## Finetune on VQA data
### train
```bash
python run.py with num_gpus=1 \
    num_nodes=1 \
    task_finetune_vqa_clip_bert \
    per_gpu_batchsize=8 \
    load_path=result/official_released/meter_clip16_288_roberta_pretrain.ckpt \
    clip16 text_roberta \
    image_size=224 \
    nce=True \
    test_only=False \
    seed=0 \
    exp_name=finetune_vqa_cmcl 
```

### test
```bash
python run.py with num_gpus=1 \
    num_nodes=1 \
    task_finetune_vqa_clip_bert \
    per_gpu_batchsize=8 \
    load_path=path/to/finetuned/ckpt \
    clip16 text_roberta \
    image_size=224 \
    nce=True \
    test_only=True \
    seed=0 \
    exp_name=finetune_vqa_cmcl 
```
