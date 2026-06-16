import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np
import random
import csv


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".npy"}


@dataclass(frozen=True)
class Sample:
    img: Path
    # label may not exist (negative sample), but we still generate an empty txt
    # under the fold to maintain one-to-one correspondence
    label: Path
    key: str  # Group key — samples with the same key won't be split across folds


def default_group_key(stem: str, group_by: str) -> str:
    """
    Generate a group key from the filename stem.

    Filename format: E0001013_20091215_slice140_y32_x32
    - none: each image is a separate sample (key=stem)
    - case: first underscore segment as case (key=E0001013)
    - case_date: first two segments as case+date (key=E0001013_20091215)
    """
    parts = stem.split('_')
    name = parts[0]
    for part in parts[1:]:
        if 'slice' in part:
            break
        name = name + '_' + f'{part}'
    return name
    # if group_by == "none":
    #     return stem
    # parts = stem.split("_")
    # if group_by == "case":
    #     return parts[0] if parts else stem
    # if group_by == "case_date":
    #     return "_".join(parts[:2]) if len(parts) >= 2 else stem
    # raise ValueError(f"Unknown group_by: {group_by}")


def collect_samples(image_dirs: Sequence[Path], label_dirs: Sequence[Path], group_by: str, neg_ratio: float = 0.1) -> List[Sample]:
    if len(image_dirs) != len(label_dirs):
        raise ValueError("image_dirs and label_dirs must have the same length")

    pos_samples: List[Sample] = []
    neg_samples: List[Sample] = []

    for img_dir, lab_dir in zip(image_dirs, label_dirs):
        if not img_dir.is_dir() or not lab_dir.is_dir():
            continue

        for p in img_dir.iterdir():
            if p.suffix.lower() not in IMAGE_EXTS:
                continue

            stem = p.stem
            label_file = lab_dir / f"{stem}.txt"
            key = default_group_key(stem, group_by=group_by)
            sample = Sample(img=p, label=label_file, key=key)

            # Positive if label file exists and is non-empty; otherwise negative
            if label_file.exists() and label_file.stat().st_size > 0:
                pos_samples.append(sample)
            else:
                neg_samples.append(sample)

    # --- Positive/Negative Ratio Control ---
    num_pos = len(pos_samples)
    if num_pos == 0:
        print("Warning: no positive samples found! Returning all negative samples.")
        return neg_samples

    # Target negative count: neg / (pos + neg) = ratio  =>  neg = (pos * ratio) / (1 - ratio)
    # For a 9:1 ratio, ratio = 0.1
    target_neg_count = int((num_pos * neg_ratio) / (1 - neg_ratio))

    if len(neg_samples) > target_neg_count:
        print(f"Positive: {num_pos}, original negative: {len(neg_samples)} -> downsampling to: {target_neg_count}")
        random.seed(42)  # Reproducible downsampling
        neg_samples = random.sample(neg_samples, target_neg_count)
    else:
        print(f"Positive: {num_pos}, insufficient negatives, keeping all: {len(neg_samples)}")

    all_samples = pos_samples + neg_samples
    # Stable sort ensures consistent key grouping in K-Fold
    all_samples.sort(key=lambda s: str(s.img))
    return all_samples


def kfold_split_by_group(
    samples: Sequence[Sample],
    k: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    """
    Return k (train_indices, val_indices) tuples.
    Split by group key: samples with the same key won't be split across folds.
    """
    # group -> sample indices
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.key, []).append(i)

    keys = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    folds: List[List[str]] = [[] for _ in range(k)]
    for idx, key in enumerate(keys):
        folds[idx % k].append(key)

    splits: List[Tuple[List[int], List[int]]] = []

    for fold_i in range(k):
        val_keys = set(folds[fold_i])
        val_idx: List[int] = []
        train_idx: List[int] = []
        for key, idxs in groups.items():
            (val_idx if key in val_keys else train_idx).extend(idxs)
        val_idx.sort()
        train_idx.sort()
        splits.append((train_idx, val_idx))
    return splits, folds


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """
    mode: symlink | hardlink | copy
    """
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link_mode: {mode}")


def write_empty_label(dst: Path) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    dst.write_text("")


def write_yaml(path: Path, train_images: Path, val_images: Path, names: List[str]) -> None:
    # Ultralytics supports absolute paths
    nc = len(names)
    content = (
        f"train: {train_images}\n"
        f"val: {val_images}\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    path.write_text(content)


def main(args) -> None:

    yolo_root = Path(args.yolo_root).resolve()
    out_root = (yolo_root / args.out_dirname).resolve()
    ensure_dir(out_root)

    splits = [s.strip() for s in args.use_splits.split(",") if s.strip()]
    image_dirs: List[Path] = []
    label_dirs: List[Path] = []
    for s in splits:
        image_dirs.append(yolo_root / s / "images")
        label_dirs.append(yolo_root / s / "labels")

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if not names:
        raise ValueError("--names cannot be empty")
    # Merge images and labels from train/valid
    samples = collect_samples(
        image_dirs=image_dirs,
        label_dirs=label_dirs,
        group_by=args.group_by,
        neg_ratio=args.neg_ratio
    )
    if not samples:
        raise RuntimeError("No image samples collected — check the paths")

    splits_idx, key_to_fold = kfold_split_by_group(samples=samples, k=args.k, seed=args.seed)
    csv_path = yolo_root / args.out_dirname / "fold_assignment.csv"
    if not csv_path.exists():
        header = ['val_fold', 'patient_key']
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for fold_idx, keys in enumerate(key_to_fold):
                for key in keys:
                    writer.writerow([fold_idx, key])
        print(f"Validation fold assignment saved to: {csv_path}")

    for fold_i, (train_idx, val_idx) in enumerate(splits_idx):
        # Only save one fold at a time
        # if fold_i != args.save_fold:
        #     continue
        print("-"*20 + f"constructing fold-{fold_i}" + "-"*20)
        fold_dir = out_root / f"fold{fold_i}"
        train_img_dir = fold_dir / "images" / "train"
        val_img_dir = fold_dir / "images" / "val"
        train_lab_dir = fold_dir / "labels" / "train"
        val_lab_dir = fold_dir / "labels" / "val"

        for d in [train_img_dir, val_img_dir, train_lab_dir, val_lab_dir]:
            ensure_dir(d)

        # Generate links / copies
        for count, idx in enumerate(tqdm(train_idx, desc="processing train data")):
            if count % 10000 == 0:
                os.sync()
            s = samples[idx]
            link_or_copy(s.img, train_img_dir / s.img.name, mode=args.link_mode)
            if s.label.exists():
                link_or_copy(s.label, train_lab_dir / s.label.name, mode=args.link_mode)
            # else:
                # write_empty_label(train_lab_dir / s.label.name)

        for count, idx in enumerate(tqdm(val_idx, desc="processing valid data")):
            if count % 10000 == 0:
                os.sync()
            s = samples[idx]
            link_or_copy(s.img, val_img_dir / s.img.name, mode=args.link_mode)
            if s.label.exists():
                link_or_copy(s.label, val_lab_dir / s.label.name, mode=args.link_mode)
            # else:
            #     write_empty_label(val_lab_dir / s.label.name)

        # Write yaml per fold
        yaml_path = out_root / f"fold{fold_i}.yaml"
        write_yaml(
            yaml_path,
            train_images=train_img_dir,
            val_images=val_img_dir,
            names=names,
        )

        print(
            f"[fold{fold_i}] train={len(train_idx)} val={len(val_idx)} "
            f"yaml={yaml_path}"
        )


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Generate YOLO k-fold dataset (Ultralytics).")
    ap.add_argument(
        "--yolo_root",
        default="/data/yolo_dataset_249",
        help="YOLO dataset root directory, containing train/ valid etc.",
    )
    ap.add_argument(
        "--neg_ratio",
        type=float,
        default=0,
        help="Negative sample ratio in total dataset (0.1 = pos:neg = 9:1)",
    )
    ap.add_argument("--k", type=int, default=5, help="Number of folds")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for fold assignment")
    ap.add_argument(
        "--group_by",
        choices=["none", "case", "case_date"],
        default="case_date",
        help="Group by filename prefix to prevent same-case leakage across folds",
    )
    ap.add_argument(
        "--link_mode",
        choices=["symlink", "hardlink", "copy"],
        default="hardlink",
        help="Method for generating fold data: symlink / hardlink / copy",
    )
    ap.add_argument(
        "--out_dirname",
        default="Kfold_neg_0",
        help="Output directory name, created under yolo_root (e.g. yolo_dataset/kfold/)",
    )
    ap.add_argument(
        "--use_splits",
        default="All",
        help="Split directories for k-fold, comma-separated (options: train, valid; All). "
             "Each directory must contain images/ and labels/ subdirectories.",
    )
    ap.add_argument(
        "--save_fold",
        type=int,
        default=4,
        help="Fold index [0-4] to save (saving all 5 folds at once consumes too many resources)",
    )
    ap.add_argument(
        "--names",
        default="mucus",
        help="Class names, comma-separated. Default: single class 'mucus' (class=0)",
    )
    args = ap.parse_args()
    main(args)
