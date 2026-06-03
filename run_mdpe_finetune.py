"""
Fine-tune VideoMAE on MDPE face anti-spoofing dataset (video-level).
Architecture: ViT encoder (per-clip) → cat features → per-clip-count Linear head

Usage:
    python run_mdpe_finetune.py \
        --data_path F:/MDPEv1/cash/cache_cs224_fi20_afFalse/merged_foldf0_fi20.pt \
        --finetune path/to/pretrained_videomae_base.pth \
        --output_dir work_dir/mdpe
"""

import argparse
import collections
import datetime
import os
import time
from functools import partial
from math import cos, pi

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma
from tqdm import tqdm
import wandb

import models  # noqa: F401 - register models
from dataset_mdpe import MDPEVideoDataset, video_collate_fn
from optim_factory import LayerDecayValueAssigner, create_optimizer


def get_args():
    p = argparse.ArgumentParser("VideoMAE video-level fine-tuning on MDPE")

    # Data
    p.add_argument("--data_path", type=str,
                   default="F:/MDPEv1/cash/cache_cs224_fi20_afFalse/merged_foldf0_fi20.pt")
    p.add_argument("--clip_len", type=int, default=16)

    # Model
    p.add_argument("--model", default="vit_base_patch16_224", type=str)
    p.add_argument("--tubelet_size", type=int, default=2)
    p.add_argument("--input_size", default=224, type=int)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--attn_drop_rate", type=float, default=0.0)
    p.add_argument("--drop_path", type=float, default=0.3)
    p.add_argument("--head_drop_rate", type=float, default=0.5)
    p.add_argument("--nb_classes", type=int, default=2)
    p.add_argument("--use_mean_pooling", action="store_true", default=True)
    p.add_argument("--with_checkpoint", action="store_true", default=False)
    p.add_argument("--init_scale", default=0.001, type=float)

    # Pre-trained weights
    p.add_argument("--finetune", default="vit_b_k710_dl_from_giant.pth", type=str)
    p.add_argument("--model_key", default="model|module", type=str)

    # Optimizer
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("--update_freq", default=1, type=int)
    p.add_argument("--opt", default="adamw", type=str)
    p.add_argument("--opt_eps", default=1e-8, type=float)
    p.add_argument("--opt_betas", default=None, type=float, nargs="+")
    p.add_argument("--clip_grad", type=float, default=None)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr", default=5e-5, type=float)
    p.add_argument("--layer_decay", type=float, default=0.9)
    p.add_argument("--min_lr", default=1e-6, type=float)
    p.add_argument("--warmup_lr", default=1e-8, type=float)
    p.add_argument("--warmup_epochs", default=5, type=int)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--weight_decay_end", default=None, type=float)

    # Augmentation
    p.add_argument("--aa", type=str, default="rand-m0-n2-mstd0.5-inc1")
    p.add_argument("--train_interpolation", type=str, default="bicubic")
    p.add_argument("--smoothing", type=float, default=0.1)

    # EMA
    p.add_argument("--model_ema", action="store_true", default=True)
    p.add_argument("--model_ema_decay", default=0.9999, type=float)

    # Output
    p.add_argument("--output_dir", default="work_dir/mdpe", type=str)

    # Device
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--num_workers", default=0, type=int)

    # Resume
    p.add_argument("--resume", default="", type=str,
                   help="Path to last checkpoint to resume training from")

    # WandB
    p.add_argument("--wandb_project", default="videomae-mdpe", type=str)
    p.add_argument("--wandb_entity", default="", type=str)
    p.add_argument("--wandb_name", default="", type=str)
    p.add_argument("--wandb_id", default="", type=str)
    p.add_argument("--no_wandb", action="store_true", default=False)

    return p.parse_args()


def cosine_scheduler(base_val, final_val, epochs, niter_per_ep, warmup_epochs=0):
    schedule = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            ratio = (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            ratio = final_val + 0.5 * (base_val - final_val) * (1 + cos(pi * progress))
        schedule.extend([ratio] * niter_per_ep)
    return schedule


def load_pretrained(model, args):
    if not args.finetune:
        return

    print(f"Loading pre-trained weights from {args.finetune}")
    checkpoint = torch.load(args.finetune, map_location="cpu")

    ckpt = checkpoint
    for key in args.model_key.split("|"):
        if key in checkpoint:
            ckpt = checkpoint[key]
            print(f"  Using key: {key}")
            break

    new_ckpt = collections.OrderedDict()
    for k, v in ckpt.items():
        k = k.replace("_orig_mod.", "").replace("backbone.", "").replace("encoder.", "")
        new_ckpt[k] = v

    # Remove head if shape mismatch
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in new_ckpt and new_ckpt[k].shape != state_dict[k].shape:
            print(f"  Removing {k}: {new_ckpt[k].shape} != {state_dict[k].shape}")
            del new_ckpt[k]

    # Remove video_heads (new, not in pretrained checkpoint)
    for k in list(new_ckpt.keys()):
        if k.startswith("video_heads."):
            del new_ckpt[k]

    # Interpolate positional embedding
    if "pos_embed" in new_ckpt:
        pos_ckpt = new_ckpt["pos_embed"]
        tubelet_size = model.patch_embed.tubelet_size
        num_patches = model.patch_embed.num_patches
        embedding_size = pos_ckpt.shape[-1]

        orig_T = 16 // tubelet_size
        new_T = args.clip_len // tubelet_size
        orig_S = int((pos_ckpt.shape[1] // orig_T) ** 0.5)
        new_S = int((num_patches // new_T) ** 0.5)

        if orig_S != new_S or orig_T != new_T:
            print(f"  Interpolating pos_embed: spatial {orig_S}->{new_S}, temporal {orig_T}->{new_T}")
            pos = pos_ckpt.reshape(1, orig_T, orig_S, orig_S, embedding_size)
            pos = pos.reshape(-1, orig_S, orig_S, embedding_size).permute(0, 3, 1, 2)
            pos = torch.nn.functional.interpolate(
                pos, size=(new_S, new_S), mode="bicubic", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, orig_T, new_S, new_S, embedding_size)
            pos = pos.reshape(-1, new_S * new_S, embedding_size).permute(0, 2, 1)
            pos = torch.nn.functional.interpolate(pos, size=new_T, mode="linear")
            pos = pos.permute(0, 2, 1).reshape(1, new_T * new_S * new_S, embedding_size)
            new_ckpt["pos_embed"] = pos

    msg = model.load_state_dict(new_ckpt, strict=False)
    print(f"  Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    lr_schedule, wd_schedule, args, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, (clips, labels, num_clips_list) in enumerate(pbar):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # LR & WD schedule
        it = epoch * len(loader) + step
        if it < len(lr_schedule):
            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule[it] * pg.get("lr_scale", 1.0)
        if it < len(wd_schedule):
            for pg in optimizer.param_groups:
                if pg["weight_decay"] > 0:
                    pg["weight_decay"] = wd_schedule[it]

        with torch.cuda.amp.autocast():
            outputs = model(clips, num_clips=num_clips_list)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if not args.no_wandb:
            wandb.log({
                "train/loss_step": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/wd": optimizer.param_groups[0].get("weight_decay", 0),
            }, step=epoch * len(loader) + step)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for clips, labels, num_clips_list in tqdm(loader, desc="Val"):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(clips, num_clips=num_clips_list)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- WandB ----
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name or None,
            id=args.wandb_id or None,
            config=vars(args),
            resume="allow" if args.wandb_id else None,
        )

    # ---- Load split info ----
    print("Loading train/test split ...")
    data = torch.load(args.data_path, map_location="cpu")
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    del data

    # ---- Dataset ----
    dataset_train = MDPEVideoDataset(
        args.data_path, clip_len=args.clip_len, mode="train",
        train_idx=train_idx, test_idx=test_idx, args=args)
    dataset_val = MDPEVideoDataset(
        args.data_path, clip_len=args.clip_len, mode="val",
        train_idx=train_idx, test_idx=test_idx, args=args)

    loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=video_collate_fn,
        persistent_workers=True)
    loader_val = DataLoader(
        dataset_val, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=True)

    # ---- Model ----
    all_clip_counts = sorted(dataset_train.clip_counts | dataset_val.clip_counts)
    print(f"All clip counts across dataset: {all_clip_counts}")

    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.clip_len,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        head_drop_rate=args.head_drop_rate,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        with_cp=args.with_checkpoint,
        clip_counts=all_clip_counts,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, params: {n_params / 1e6:.1f}M")
    print(f"Video heads: {all_clip_counts}")

    load_pretrained(model, args)
    model.to(device)

    # EMA
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay)

    # ---- Optimizer with layer-wise LR decay ----
    num_layers = model.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        print(f"Layer decay: {assigner.values[:3]}...{assigner.values[-2:]}")
    else:
        assigner = None

    skip_wd = model.no_weight_decay()
    optimizer = create_optimizer(
        args, model,
        skip_list=skip_wd,
        get_num_layer=assigner.get_layer_id if assigner else None,
        get_layer_scale=assigner.get_scale if assigner else None)

    scaler = torch.cuda.amp.GradScaler()
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # ---- Schedules ----
    niter = len(loader_train)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.epochs, niter, args.warmup_epochs)
    wd_end = args.weight_decay_end if args.weight_decay_end else args.weight_decay
    wd_schedule = cosine_scheduler(args.weight_decay, wd_end, args.epochs, niter)

    # ---- Resume ----
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if model_ema is not None and "model_ema" in ckpt:
            model_ema.ema.load_state_dict(ckpt["model_ema"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"  Resume from epoch {start_epoch}, best_acc={best_acc:.4f}")

    # ---- Train ----
    print(f"\nStart training: epoch {start_epoch}->{args.epochs}, {niter} iters/epoch")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, loader_train, optimizer, criterion, device, epoch,
            lr_schedule, wd_schedule, args, scaler)

        val_stats = validate(model, loader_val, criterion, device)

        print(f"Epoch {epoch}: train_loss={train_stats['loss']:.4f} "
              f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}")

        # Save best
        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            save_ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
            }
            if model_ema is not None:
                save_ckpt["model_ema"] = model_ema.ema.state_dict()
            torch.save(save_ckpt,
                       os.path.join(args.output_dir, "best_checkpoint.pth"))
            print(f"  -> Best: {best_acc:.4f}")

        # Save last
        last_path = os.path.join(args.output_dir, "last_checkpoint.pth")
        save_ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if model_ema is not None:
            save_ckpt["model_ema"] = model_ema.ema.state_dict()
        torch.save(save_ckpt, last_path)

        # EMA update
        if model_ema is not None:
            model_ema.update(model)

        if not args.no_wandb:
            wandb.log({
                "train/loss": train_stats["loss"],
                "train/acc": train_stats["acc"],
                "val/loss": val_stats["loss"],
                "val/acc": val_stats["acc"],
                "val/best_acc": best_acc,
                "epoch": epoch,
            }, step=epoch)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Done in {elapsed}. Best val acc: {best_acc:.4f}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
