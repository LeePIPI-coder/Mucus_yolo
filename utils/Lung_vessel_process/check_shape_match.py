#!/usr/bin/env python3
"""Check if image.nii.gz and vessle_name.nii.gz shapes match in each subdirectory."""

import argparse
from pathlib import Path

import nibabel as nib


def main():
    parser = argparse.ArgumentParser(
        description="比对各子目录下 image.nii.gz 和 vessle_name.nii.gz 的形状是否一致."
    )
    parser.add_argument("root_dir", nargs="?", default="/data/Mucus_data",
                        type=str, help="Root directory containing subdirectories")
    args = parser.parse_args()

    root = Path(args.root_dir)
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())

    same = []
    diff = []
    missing = []

    for subdir in subdirs:
        img_path = subdir / "image.nii.gz"
        ves_path = subdir / "vessle_name.nii.gz"

        if not img_path.exists() or not ves_path.exists():
            missing.append(subdir.name)
            continue

        img_shape = nib.load(img_path).shape
        ves_shape = nib.load(ves_path).shape

        if img_shape == ves_shape:
            same.append(subdir.name)
        else:
            diff.append((subdir.name, img_shape, ves_shape))

    print(f"形状相同: {len(same)}")
    for n in same:
        print(f"  = {n}")

    print(f"\n形状不同: {len(diff)}")
    for n, ishp, vshp in diff:
        print(f"  ≠ {n}  image={ishp}  vessel={vshp}")

    print(f"\n缺少文件: {len(missing)}")
    for n in missing:
        print(f"  ? {n}")

    print(f"\n总计: 相同={len(same)}, 不同={len(diff)}, 缺少={len(missing)}")


if __name__ == "__main__":
    main()
