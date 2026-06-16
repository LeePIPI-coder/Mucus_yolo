#!/usr/bin/env python3
"""Match .nii.gz filenames to subdirectory names and copy files into them."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Copy .nii.gz files into matching subdirectories by name."
        " 将niigz文件按名称匹配到对应子目录并复制."
    )
    parser.add_argument("src_dir", nargs="?", default="/data/LungVes_mask_reversed", type=str, help="Directory containing .nii.gz files")
    parser.add_argument("dst_dir", nargs="?", default="/data/Mucus_data", type=str, help="Directory containing subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no copy")
    args = parser.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)

    nii_files = sorted(src.glob("*.nii.gz"))
    subdir_names = {d.name for d in dst.iterdir() if d.is_dir()}

    matched = 0
    unmatched_src = []

    for nii_path in nii_files:
        name = nii_path.name.replace(".nii.gz", "")  # e.g. "E0001009_20091120"
        vessle_name = "vessle_name.nii.gz"
        if name in subdir_names:
            target = dst / name / vessle_name
            if not args.dry_run:
                shutil.copy2(nii_path, target)
            print(f"✓ {name}")
            matched += 1
        else:
            unmatched_src.append(name)

    # Report
    print(f"\n成功复制: {matched}")
    if unmatched_src:
        print(f"源文件无对应目录 ({len(unmatched_src)}):")
        for n in unmatched_src:
            print(f"  - {n}")

    # Also report subdirs with no matching .nii.gz
    nii_stems = {p.name.replace(".nii.gz", "") for p in nii_files}
    unmatched_dst = sorted(subdir_names - nii_stems)
    if unmatched_dst:
        print(f"目录无对应源文件 ({len(unmatched_dst)}):")
        for n in unmatched_dst:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
