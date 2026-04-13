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
    # label 可能不存在（负样本），但我们仍会在 fold 下生成一个空 txt 以保持一一对应
    label: Path
    key: str  # 用于分组（可选），避免同一主体泄漏到不同 fold


def default_group_key(stem: str, group_by: str) -> str:
    """
    根据文件名生成分组 key。
    你的文件名形如：E0001013_20091215_slice140_y32_x32
    - none: 每张图单独作为一个样本（key=stem）
    - case: 以第一个下划线前缀作为 case（key=E0001013）
    - case_date: 以前两段作为 case+date（key=E0001013_20091215）
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
    # raise ValueError(f"未知 group_by: {group_by}")


def collect_samples(image_dirs: Sequence[Path], label_dirs: Sequence[Path], group_by: str, neg_ratio: float = 0.1) -> List[Sample]:
    if len(image_dirs) != len(label_dirs):
        raise ValueError("image_dirs 和 label_dirs 长度必须一致")

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

            # 判断正负样本：标签文件存在且非空为正，否则为负
            if label_file.exists() and label_file.stat().st_size > 0:
                pos_samples.append(sample)
            else:
                neg_samples.append(sample)

    # --- 正负样本比例控制逻辑 ---
    num_pos = len(pos_samples)
    if num_pos == 0:
        print("警告：未发现正样本！将返回所有负样本。")
        return neg_samples
    
    # 计算目标负样本数量。公式：neg / (pos + neg) = ratio  =>  neg = (pos * ratio) / (1 - ratio)
    # 对于 9:1 比例，ratio = 0.1
    target_neg_count = int((num_pos * neg_ratio) / (1 - neg_ratio))
    
    if len(neg_samples) > target_neg_count:
        print(f"正样本: {num_pos}, 原始负样本: {len(neg_samples)} -> 下采样至: {target_neg_count}")
        random.seed(42) # 保证下采样可复现
        neg_samples = random.sample(neg_samples, target_neg_count)
    else:
        print(f"正样本: {num_pos}, 负样本充足度不足，保留全部负样本: {len(neg_samples)}")

    all_samples = pos_samples + neg_samples
    # 稳定排序，保证后续 K-Fold 的 key 分组逻辑一致
    all_samples.sort(key=lambda s: str(s.img))
    return all_samples


def kfold_split_by_group(
    samples: Sequence[Sample],
    k: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    """
    返回 k 个 (train_indices, val_indices)
    按 group key 做 split：同一个 key 的样本不会被拆到不同 fold。
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
        # 稳定排序
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
        # 大数据不推荐，但提供兜底
        import shutil

        shutil.copy2(src, dst)
    else:
        raise ValueError(f"未知 link_mode: {mode}")


def write_empty_label(dst: Path) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    dst.write_text("")


def write_yaml(path: Path, train_images: Path, val_images: Path, names: List[str]) -> None:
    # Ultralytics 支持绝对路径；这里写绝对路径更省心
    nc = len(names)
    content = (
        f"train: {train_images}\n"
        f"val: {val_images}\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    path.write_text(content)


def main(args) -> None:

    # 获取需要分折数据的地址
    yolo_root = Path(args.yolo_root).resolve()
    # 分折后保存的数据地址
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
        raise ValueError("--names 不能为空")
    # 将train、valid中的images、label进行合并
    samples = collect_samples(
        image_dirs=image_dirs, 
        label_dirs=label_dirs, 
        group_by=args.group_by,
        neg_ratio=args.neg_ratio
    )
    if not samples:
        raise RuntimeError("未收集到任何图片样本，请检查路径是否正确")

    splits_idx, key_to_fold = kfold_split_by_group(samples=samples, k=args.k, seed=args.seed)
    csv_path = yolo_root / args.out_dirname / "fold_assignment.csv"    
    if not csv_path.exists():
        header = ['val_fold','patient_key']
        # 写入文件
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # 写入第一行表头

            # 写入数据
            for fold_idx, keys in enumerate(key_to_fold):
                for key in keys:
                    writer.writerow([fold_idx,key])
        print(f"已将验证集分配记录保存至: {csv_path}")      
    
    for fold_i, (train_idx, val_idx) in enumerate(splits_idx):
        # 每次只保存save_fold一折数据
        # if fold_i != args.save_fold:
        #     continue
        print("-"*20 + f"constructing fold-{fold_i}" + "-"*20)
        fold_dir = out_root / f"fold{fold_i}"
        train_img_dir = fold_dir / "images" / "train"
        val_img_dir = fold_dir / "images" / "val"
        train_lab_dir = fold_dir / "labels" / "train"
        val_lab_dir = fold_dir / "labels" / "val"

        # 创建目录
        for d in [train_img_dir, val_img_dir, train_lab_dir, val_lab_dir]:
            ensure_dir(d)

        # 生成链接/复制
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

        # 写每折的 yaml
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
        help="yolo_dataset 根目录，包含 train/ valid等",
    )
    ap.add_argument(
        "--neg_ratio", 
        type=float, 
        default=0, 
        help="负样本占总数据集的比例 (0.1 表示 正:负 = 9:1)"
    )
    ap.add_argument("--k", type=int, default=5, help="折数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（用于 fold 分配）")
    ap.add_argument(
        "--group_by",
        choices=["none", "case", "case_date"],
        default="case_date",
        help="按文件名前缀分组，避免同一case泄漏到不同 fold",
    )
    ap.add_argument(
        "--link_mode",
        choices=["symlink", "hardlink", "copy"],
        default="hardlink",
        help="生成 fold 时采用的方式：软链接/硬链接/复制",
    )
    ap.add_argument(
        "--out_dirname",
        default="Kfold_neg_0",
        help="输出目录名，会创建在 yolo-root 下，例如 yolo_dataset/kfold/",
    )
    ap.add_argument(
        "--use_splits",
        default="All",
        help="参与 k-fold 的 split目录，用逗号分隔(可选参数:train,valid; All), 目录下要包含images,labels目录",
    )
    ap.add_argument(
        "--save_fold",
        type=int,
        default=4,
        help="参数[0,1,2,3,4];需要保存的折数(因直接保存5折图像，系统会占用大量资源，因此每次只保存一折数据)",
    )
    ap.add_argument(
        "--names",
        default="mucus",
        help="类别名，用逗号分隔。默认单类 mucus（对应 class=0）",
    )
    args = ap.parse_args()
    main(args)
