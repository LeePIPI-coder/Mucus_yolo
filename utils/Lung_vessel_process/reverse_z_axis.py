#!/usr/bin/env python3
"""Reverse the Z-axis of all .nii.gz files in a directory and save."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def reverse_z_axis(input_path: Path, output_path: Path | None = None) -> None:
    """Load a NIfTI file, reverse its Z-axis, and save."""
    img = nib.load(input_path)
    data = img.get_fdata()
    data_rev = np.flip(data, axis=2)  # Z is the 3rd axis (0-indexed: 2)
    new_img = nib.Nifti1Image(data_rev, img.affine, img.header)
    out = output_path or input_path
    nib.save(new_img, str(out))
    print(f"Processed: {input_path.name} → {out}")


def main():
    parser = argparse.ArgumentParser(description="Reverse Z-axis of NIfTI files.")
    parser.add_argument("input_dir", type=str, help="Directory containing .nii.gz files")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: overwrite in place)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    nii_files = sorted(input_dir.glob("*.nii.gz"))
    if not nii_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for nii_path in nii_files:
        out_path = output_dir / nii_path.name if output_dir else None
        reverse_z_axis(nii_path, out_path)

    print(f"Done. Processed {len(nii_files)} file(s).")


if __name__ == "__main__":
    main()
