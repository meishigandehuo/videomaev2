# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoMAEv2 is the official implementation of the CVPR 2023 paper "Scaling Video Masked Autoencoders with Dual Masking". It introduces a self-supervised video pre-training framework where the encoder applies tube masking (90%) and the decoder applies running-cell masking (50%) — together reconstructing only ~5% of total patches. Supports ViT-S/B/L/H/G models up to 1.4B parameters.

**Critical dependency**: `timm==0.4.12` is pinned. All models register via `timm.create_model()`.

## Commands

### Pre-training
```bash
python -u run_mae_pretraining.py \
    --data_path data/hybrid_train.csv \
    --mask_type tube --mask_ratio 0.9 \
    --decoder_mask_type run_cell --decoder_mask_ratio 0.5 \
    --model pretrain_videomae_giant_patch14_224 \
    --decoder_depth 4 --batch_size 32 \
    --with_checkpoint --num_frames 16 --sampling_rate 4 \
    --num_sample 4 --opt adamw --lr 6e-4 --clip_grad 0.02 \
    --opt_betas 0.9 0.95 --warmup_epochs 30 --epochs 300 \
    --output_dir work_dir/output
```

### Fine-tuning
```bash
python run_class_finetuning.py \
    --model vit_giant_patch14_224 \
    --data_set Kinetics-710 --nb_classes 710 \
    --data_path data/k710 --finetune model_zoo/pretrained.pth \
    --batch_size 3 --num_frames 16 --sampling_rate 4 \
    --num_sample 2 --opt adamw --lr 1e-3 --drop_path 0.3 \
    --clip_grad 5.0 --layer_decay 0.9 \
    --opt_betas 0.9 0.999 --weight_decay 0.1 \
    --warmup_epochs 5 --epochs 35 \
    --test_num_segment 5 --test_num_crop 3 \
    --dist_eval --enable_deepspeed
```

### Evaluation only
Add `--eval` to any fine-tuning command.

### Feature extraction (temporal action detection)
```bash
python extract_tad_feature.py \
    --data_set THUMOS14 --data_path thumos14_video \
    --model vit_giant_patch14_224 --ckpt_path model.pth
```

## Architecture

### Entry points
- `run_mae_pretraining.py` — distributed pre-training with AMP, cosine LR schedule
- `run_class_finetuning.py` — fine-tuning with DeepSpeed or DDP, layer-wise LR decay, label conversion (K710→K400/600/700)
- `extract_tad_feature.py` — per-frame feature extraction for temporal action detection

### Training engines
- `engine_for_pretraining.py` — MSE loss on dual-masked positions only (`bool_masked_pos[~decode_masked_pos]`); target = per-patch normalized pixels
- `engine_for_finetuning.py` — classification with mixup/cutmix, EMA, gradient accumulation; multi-segment multi-crop evaluation with softmax averaging

### Models (`models/`)
All models use `@register_model` from timm. `import models` at entry points triggers registration.

- `modeling_pretrain.py` — **PretrainVisionTransformer** (encoder + decoder). Encoder processes only visible patches. Decoder reconstructs masked positions using mask tokens + positional embeddings. Decoder is discarded after pre-training.
- `modeling_finetune.py` — **VisionTransformer** (encoder only). PatchEmbed via 3D conv → transformer blocks → mean pooling (no CLS token) → linear head.

**Five model variants**: S (384d/12L/6H), B (768d/12L/12H), L (1024d/24L/16H), H (1280d/32L/16H), G (1408d/40L/16H, patch_size=14).

### Masking (`dataset/masking_generator.py`)
Three strategies, each operating on `(T//tubelet_size, H//patch_size, W//patch_size)` grid:
- **RandomMaskingGenerator** — independent random masking
- **TubeMaskingGenerator** — same spatial mask across all frames (encoder)
- **RunningCellMaskingGenerator** — divides grid into 2x2 cells, rotates masked positions via circular queue (decoder)

### Data pipeline (`dataset/`)
- `pretrain_datasets.py` — `HybridVideoMAE` handles both video files (decord) and raw frames, with auto-detection. Annotation format: `path start_idx total_frames` (total_frames=-1 for video files).
- `datasets.py` — `VideoClsDataset` / `RawFrameClsDataset` for fine-tuning with RandAugment and multi-crop eval.
- `build.py` — dataset factory supporting Kinetics-400/600/700/710, SSV2, UCF101, HMDB51, Diving48, MIT.

### Key utilities
- `utils.py` — distributed init (SLURM/ITP/torchrun), cosine scheduler with warmup, AMP scaler, checkpoint save/load (AMP + DeepSpeed), TensorBoard logging
- `optim_factory.py` — 15+ optimizer types, `LayerDecayValueAssigner` (scale for layer i = `layer_decay^(num_layers+1-i)`)

## Key Design Details

- **Dual masking loss**: Only positions that are encoder-masked AND decoder-visible contribute to MSE loss. This is the core mechanism — not all masked patches are reconstructed.
- **Positional embeddings**: Fixed sinusoidal (not learnable) for both encoder and decoder.
- **Checkpoint loading** in fine-tuning handles: `_orig_mod.` prefix stripping (torch.compile), K710→K400/600/700 head conversion via JSON label maps, `backbone.`/`encoder.` prefix stripping, and spatial/temporal positional embedding interpolation for resolution changes.
- **Gradient checkpointing**: `--with_checkpoint` enables `torch.utils.checkpoint` in transformer blocks.
- **Data annotation files** are plain text, one sample per line: pre-training uses 3-column format, fine-tuning uses 2-column (video) or 3-column (frames) format.

## Scripts

- `scripts/pretrain/` — pre-training configs for s/b/h/g models
- `scripts/finetune/` — 17 fine-tuning scripts for various datasets and model sizes
- `misc/` — label mapping JSONs (k710→k400/600/700)
- `docs/` — dataset preparation (DATASET.md), pre-training (PRETRAIN.md), fine-tuning (FINETUNE.md), model zoo (MODEL_ZOO.md), TAD (TAD.md)
