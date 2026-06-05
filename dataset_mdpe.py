"""
MDPE Dataset for VideoMAE fine-tuning.
Uses the same augmentation pipeline as VideoMAEv2.
"""

import sys
import os
import types

# Bypass dataset/__init__.py which imports decord
_pkg = types.ModuleType("dataset")
_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "dataset")]
_pkg.__package__ = "dataset"
sys.modules["dataset"] = _pkg

import importlib.util


def _load_submodule(name):
    path = os.path.join(os.path.dirname(__file__), "dataset", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"dataset.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"dataset.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


video_transforms = _load_submodule("video_transforms")
volume_transforms_mod = _load_submodule("volume_transforms")
rand_augment_mod = _load_submodule("rand_augment")
transforms_mod = _load_submodule("transforms")
random_erasing_mod = _load_submodule("random_erasing")

ClipToTensor = volume_transforms_mod.ClipToTensor
RandomErasing = random_erasing_mod.RandomErasing

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


def tensor_normalize(tensor, mean, std):
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    mean = torch.tensor(mean) if isinstance(mean, list) else mean
    std = torch.tensor(std) if isinstance(std, list) else std
    return (tensor - mean) / std


def spatial_sampling(
    frames, spatial_idx=-1, min_scale=256, max_scale=320,
    crop_size=224, random_horizontal_flip=True,
    inverse_uniform_sampling=False, aspect_ratio=None, scale=None,
):
    if spatial_idx == -1:
        if aspect_ratio is not None and scale is not None:
            frames = video_transforms.random_resized_crop(
                images=frames, target_height=crop_size, target_width=crop_size,
                scale=scale, ratio=aspect_ratio)
        else:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                frames, min_size=min_scale, max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling)
            frames, _ = video_transforms.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale)
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _aug_frame(buffer, crop_size=224, aa="rand-m7-n4-mstd0.5-inc1",
               train_interpolation="bicubic"):
    aug_transform = video_transforms.create_random_augment(
        input_size=(crop_size, crop_size),
        auto_augment=aa,
        interpolation=train_interpolation,
    )
    buffer = [transforms.ToPILImage()(frame) for frame in buffer]
    buffer = aug_transform(buffer)
    buffer = [transforms.ToTensor()(img) for img in buffer]
    buffer = torch.stack(buffer).permute(0, 2, 3, 1)
    buffer = tensor_normalize(buffer, IMAGENET_MEAN, IMAGENET_STD)
    buffer = buffer.permute(3, 0, 1, 2)
    buffer, _ = video_transforms.horizontal_flip(0.5, buffer)
    return buffer


def _val_frame(buffer, crop_size=224):
    transform = video_transforms.Compose([
        video_transforms.Resize(224, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(crop_size, crop_size)),
        ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(buffer)


class MDPEVideoDataset(Dataset):
    """
    Iterates over videos, returns all clips of a video together.
    Each sample = all clips tensors + label + num_clips.
    """

    def __init__(self, data_path, clip_len=16, mode="train",
                 train_idx=None, test_idx=None, args=None,
                 cached_data=None):
        self.clip_len = clip_len
        self.mode = mode

        if args is not None:
            self.aa = getattr(args, "aa", "rand-m7-n4-mstd0.5-inc1")
            self.train_interpolation = getattr(args, "train_interpolation", "bicubic")
        else:
            self.aa = "rand-m7-n4-mstd0.5-inc1"
            self.train_interpolation = "bicubic"

        if cached_data is not None:
            self.videos = cached_data["videos"]
            self.labels = cached_data["labels"]
            self.frame_counts = cached_data["frame_counts"]
        else:
            print(f"Loading {data_path} ...")
            data = torch.load(data_path, map_location="cpu")
            self.videos = data["videos"]
            self.labels = data["labels"]
            self.frame_counts = data["frame_counts"]

        video_indices = train_idx if mode == "train" else test_idx

        self.video_list = []
        self.clip_counts = set()
        for i in video_indices:
            T = self.frame_counts[i]
            n = T // clip_len
            if n > 0:
                self.video_list.append((i, n))
                self.clip_counts.add(n)

        total_clips = sum(n for _, n in self.video_list)
        label_counts = [0, 0]
        for i, n in self.video_list:
            label_counts[int(self.labels[i])] += n
        print(f"[{mode}] {len(self.video_list)} videos, {total_clips} total clips, "
              f"Real(0): {label_counts[0]}, Fake(1): {label_counts[1]}")
        print(f"[{mode}] Unique clip counts: {sorted(self.clip_counts)}")

    def __len__(self):
        return len(self.video_list)

    def _to_uint8_frames(self, clip):
        clip = clip.permute(0, 2, 3, 1)
        clip = (clip * 255).clamp(0, 255).to(torch.uint8)
        return [frame.numpy() for frame in clip]

    def __getitem__(self, idx):
        video_idx, num_clips = self.video_list[idx]
        video = self.videos[video_idx]
        label = int(self.labels[video_idx])

        clips = []
        for c in range(num_clips):
            start = c * self.clip_len
            clip = video[start: start + self.clip_len]
            frames = self._to_uint8_frames(clip)

            if self.mode == "train":
                clips.append(_aug_frame(frames, aa=self.aa,
                                        train_interpolation=self.train_interpolation))
            else:
                clips.append(_val_frame(frames))

        clips = torch.stack(clips)
        return clips, label, num_clips


def video_collate_fn(batch):
    """Flatten all clips, keep track of num_clips per video."""
    all_clips = []
    labels = []
    num_clips_list = []

    for clips, label, n in batch:
        all_clips.append(clips)
        labels.append(label)
        num_clips_list.append(n)

    all_clips = torch.cat(all_clips, dim=0)
    return all_clips, torch.tensor(labels), num_clips_list
