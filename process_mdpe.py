"""
Process MDPE dataset from merged_foldf0_fi20.pt
- Load pre-cached video tensors (T, 3, 224, 224)
- Split into consecutive 16-frame clips
- Discard videos with < 16 frames
- Discard remainder frames at the end of each video
- Save as a new dataset for VideoMAE fine-tuning
"""

import torch
import numpy as np
import os
from tqdm import tqdm

CLIP_LEN = 16

def process_dataset(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {input_path} ...")
    data = torch.load(input_path, map_location="cpu")
    videos = data["videos"]
    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    total_videos = len(videos)
    print(f"Total videos: {total_videos}")

    # Collect clips
    all_clips = []
    all_labels = []
    all_video_indices = []  # which original video each clip came from
    discard_count = 0

    for i in tqdm(range(total_videos), desc="Splitting clips"):
        video = videos[i]  # Tensor (T, 3, 224, 224)
        label = int(labels[i])
        T = video.shape[0]

        if T < CLIP_LEN:
            discard_count += 1
            continue

        num_clips = T // CLIP_LEN
        for c in range(num_clips):
            start = c * CLIP_LEN
            end = start + CLIP_LEN
            clip = video[start:end]  # (16, 3, 224, 224)
            all_clips.append(clip)
            all_labels.append(label)
            all_video_indices.append(i)

    print(f"Discarded {discard_count} videos (< {CLIP_LEN} frames)")
    print(f"Generated {len(all_clips)} clips from {len(set(all_video_indices))} videos")

    # Build new train/test split based on original video indices
    train_video_set = set(train_idx)
    test_video_set = set(test_idx)

    new_train_idx = []
    new_test_idx = []
    for clip_idx, video_idx in enumerate(all_video_indices):
        if video_idx in train_video_set:
            new_train_idx.append(clip_idx)
        elif video_idx in test_video_set:
            new_test_idx.append(clip_idx)

    print(f"Train clips: {len(new_train_idx)}, Test clips: {len(new_test_idx)}")

    # Label distribution
    train_labels = [all_labels[i] for i in new_train_idx]
    test_labels = [all_labels[i] for i in new_test_idx]
    print(f"Train - Real(0): {train_labels.count(0)}, Fake(1): {train_labels.count(1)}")
    print(f"Test  - Real(0): {test_labels.count(0)}, Fake(1): {test_labels.count(1)}")

    # Save
    output_path = os.path.join(output_dir, "mdpe_clips16.pt")
    torch.save({
        "clips": all_clips,           # list of Tensor(16, 3, 224, 224)
        "labels": all_labels,          # list of int
        "video_indices": all_video_indices,  # which original video
        "train_idx": new_train_idx,
        "test_idx": new_test_idx,
    }, output_path)

    size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"Saved to {output_path} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    input_path = "/mnt/sda2/home/lrb/dataset/cache_cs224_fi20_afFalse/merged_foldf0_fi20.pt"
    output_dir = "/mnt/sda2/home/lrb/dataset/output_dir"
    process_dataset(input_path, output_dir)
