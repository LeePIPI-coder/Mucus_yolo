#!/usr/bin/env python3
"""Replace image voxels with random values where vessel mask == 1."""

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="读取txt中名称，将对应子目录下 vessel==1 处的 image 值替换为随机数."
    )
    parser.add_argument("txt", nargs="?", default="/workspace/process.txt",
                        type=str, help="Txt file with directory names")
    parser.add_argument("--root", "-r", default="/data/Mucus_data",
                        type=str, help="Root directory containing subdirectories")
    parser.add_argument("--out-suffix", "-o", default="_modified",
                        type=str, help="Output suffix appended to image filename (空字符串则覆盖原文件)")
    parser.add_argument("--low", type=int, default=-800, help="Random range low")
    parser.add_argument("--high", type=int, default=970, help="Random range high")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.root)

    # Read names from txt
    with open(args.txt) as f:
        names = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Line format: "1	  E0001009_20101124" (number, tab, name)
            name = re.split(r'\s+', line)[-1]
            names.append(name)

    ok = []
    missing = []
    shape_mismatch = []

    for name in names:
        subdir = root / name
        img_path = subdir / "image.nii.gz"
        ves_path = subdir / "vessle_name.nii.gz"

        if not img_path.exists() or not ves_path.exists():
            missing.append(name)
            print(f"✗ {name}: 缺少文件")
            continue

        img = nib.load(img_path)
        ves = nib.load(ves_path)
        img_data = img.get_fdata()
        ves_data = ves.get_fdata()

        if img_data.shape != ves_data.shape:
            shape_mismatch.append((name, img_data.shape, ves_data.shape))
            print(f"⚠ {name}: 形状不匹配 image={img_data.shape} vessel={ves_data.shape}")
            continue

        mask = np.isclose(ves_data, 1.0)
        n_voxels = mask.sum()
        img_data[mask] = -880

        # Determine output path
        if args.out_suffix:
            out_path = img_path.parent / f"{img_path.stem}{args.out_suffix}.nii.gz"
        else:
            out_path = img_path

        new_img = nib.Nifti1Image(img_data.astype(np.float32), img.affine, img.header)
        nib.save(new_img, str(out_path))
        print(f"✓ {name}: {n_voxels} voxels replaced → {out_path.name}")
        ok.append(name)

    print(f"\n完成: ok={len(ok)}, missing={len(missing)}, shape_mismatch={len(shape_mismatch)}")


if __name__ == "__main__":
    main()
