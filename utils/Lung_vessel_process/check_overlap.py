#!/usr/bin/env python3
"""
Check overlap between vessle_name.nii.gz and mask.nii.gz for each patient in process.txt.
Reports per-slice overlap voxel counts and totals.
"""

import nibabel as nib
import numpy as np
import os

DATA_DIR = "/data/Mucus_data"
PROCESS_FILE = "/workspace/process.txt"


def check_patient_overlap(patient_dir):
    """Return (overlap_voxels, overlapping_slices_dict) for one patient."""
    v_path = os.path.join(DATA_DIR, patient_dir, "vessle_name.nii.gz")
    m_path = os.path.join(DATA_DIR, patient_dir, "mask.nii.gz")

    if not os.path.exists(v_path):
        print(f"  [SKIP] Missing: {v_path}")
        return 0, {}
    if not os.path.exists(m_path):
        print(f"  [SKIP] Missing: {m_path}")
        return 0, {}

    v = nib.load(v_path).get_fdata()
    m = nib.load(m_path).get_fdata()

    v_bin = v > 0
    m_bin = m > 0

    overlap = v_bin & m_bin
    overlap_per_slice = overlap.sum(axis=(0, 1))
    slice_indices = np.where(overlap_per_slice > 0)[0]

    total_mask = int(m_bin.sum())
    total_overlap = int(overlap.sum())
    pct = (total_overlap / total_mask * 100) if total_mask > 0 else 0.0

    slices_dict = {int(s): int(overlap_per_slice[s]) for s in slice_indices}
    return total_overlap, total_mask, pct, slices_dict


def main():
    with open(PROCESS_FILE) as f:
        patients = [line.strip() for line in f if line.strip()]

    print(f"{'='*70}")
    print(f"Overlap analysis: vessle_name.nii.gz vs mask.nii.gz")
    print(f"Total patients: {len(patients)}")
    print(f"{'='*70}\n")

    grand_total_overlap = 0
    grand_total_mask = 0
    grand_total_slices = 0

    for patient in patients:
        print(f"{patient}")
        total_overlap, total_mask, pct, slices = check_patient_overlap(patient)

        grand_total_overlap += total_overlap
        grand_total_mask += total_mask
        grand_total_slices += len(slices)

        if not slices:
            print(f"  mask voxels: {total_mask}, overlapping: 0 (0.00%)\n")
            continue

        print(f"  mask (mucus) voxels: {total_mask}")
        print(f"  overlapping voxels:  {total_overlap} ({pct:.2f}%)")
        print(f"  overlapping slices:  {len(slices)}")
        print(f"  Slice details:")
        for s, count in sorted(slices.items()):
            print(f"    slice {s}: {count} voxels")
        print()

    overall_pct = (grand_total_overlap / grand_total_mask * 100) if grand_total_mask > 0 else 0.0
    print(f"{'='*70}")
    print(f"Summary")
    print(f"  Total mask (mucus) voxels:               {grand_total_mask}")
    print(f"  Total overlapping voxels:                {grand_total_overlap}")
    print(f"  Overall overlap rate:                    {overall_pct:.2f}%")
    print(f"  Total overlapping slices:                {grand_total_slices}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
