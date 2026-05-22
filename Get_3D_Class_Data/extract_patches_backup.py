"""Extract 3D CT patches for 2nd-stage classification.

Workflow per fold:
  1. Read Prediction_TP_FP_fold_x.csv
  2. Separate TP (GT) and FP candidates
  3. Gray-zone filter: discard FP with 3D IoU > 0 against any TP
  4. Stratified sample FP to achieve target GT:FP ratio
  5. Group candidates by nifti volume for efficient CT loading
  6. For each CT: load → resample to target spacing → extract patch → save

Usage:
  python extract_patches.py --gt_fp_ratio 0.5    # 1:2 (190 GT : 380 FP)
  python extract_patches.py --gt_fp_ratio 1.0    # 1:1 (190 GT : 190 FP)
  python extract_patches.py --gt_fp_ratio 0.5 --n_jitter 10 --dry_run
"""

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.ndimage import zoom as ndzoom
from tqdm import tqdm
import argparse
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_dicom_map(data_root: str) -> dict:
    """Recursively find nii.gz files under dirs whose name contains 'image', return {patient_key: nii.gz_path}."""
    data_root = Path(data_root)
    nifti_map = {}
    for nii_path in data_root.rglob("*.nii.gz"):
        if "image" not in nii_path.parent.name:
            continue
        patient_key = nii_path.name.replace(".nii.gz", "")
        nifti_map[patient_key] = str(nii_path)
    return nifti_map


def load_dicom_series(nii_path: str):
    """Load nii.gz file, return (ct_data[x,y,z], affine[4,4])."""
    img = nib.load(nii_path)
    ct_data = img.get_fdata().astype(np.float32)  # (x, y, z)
    return ct_data, img.affine


def world_to_orig_voxel(center_world, affine):
    """Convert world (mm) → original CT voxel indices [i, j, k]."""
    inv_affine = np.linalg.inv(affine)
    return nib.affines.apply_affine(inv_affine, center_world)  # → [i, j, k]


def compute_zoom(orig_spacing, target_spacing):
    """Zoom factors per axis: orig_spacing / target_spacing."""
    return np.array(orig_spacing) / np.array(target_spacing)


def compute_3d_iou(min1, max1, min2, max2):
    """Compute 3D IoU between two axis-aligned bounding boxes.

    All inputs are (3,) arrays in the same coordinate system.
    """
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_dims)
    if inter_vol == 0:
        return 0.0
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0


def extract_patch(ct_resampled, center_ijk, patch_size, jitter_range=0):
    """Crop a patch of `patch_size` around center_ijk from resampled CT.

    Parameters
    ----------
    ct_resampled : np.ndarray (3D)
    center_ijk : (ci, cj, ck) in resampled voxel coords
    patch_size : (pi, pj, pk)
    jitter_range : float, max random offset in resampled voxels (0 = no jitter)

    Returns
    -------
    patch : np.ndarray or None if center is outside volume
    meta : dict with actual center, crop bounds, etc.
    """
    ci, cj, ck = center_ijk

    # Apply jitter
    if jitter_range > 0:
        ji = np.random.uniform(-jitter_range, jitter_range)
        jj = np.random.uniform(-jitter_range, jitter_range)
        jk = np.random.uniform(-jitter_range, jitter_range)
        ci += ji; cj += jj; ck += jk

    pi, pj, pk = patch_size
    ri, rj, rk = pi // 2, pj // 2, pk // 2

    i0 = int(round(ci)) - ri
    i1 = i0 + pi
    j0 = int(round(cj)) - rj
    j1 = j0 + pj
    k0 = int(round(ck)) - rk
    k1 = k0 + pk

    shape = ct_resampled.shape  # (ni, nj, nk)

    # Clamp to volume bounds
    if i0 < 0 or j0 < 0 or k0 < 0 or i1 > shape[0] or j1 > shape[1] or k1 > shape[2]:
        return None, None

    patch = ct_resampled[i0:i1, j0:j1, k0:k1].copy()
    return patch, {"center_ijk": (ci, cj, ck), "crop_bounds": (i0, i1, j0, j1, k0, k1)}


# ---------------------------------------------------------------------------
# Main per-fold logic
# ---------------------------------------------------------------------------

def process_folds(
    data_dir: str,
    nifti_root: str,
    output_dir: str,
    gt_fp_ratio: float = 0.5,
    per_scan_cap: int = 100,
    n_jitter: int = 10,
    jitter_range: float = 4.0,
    patch_size: tuple = (64, 64, 32),
    target_spacing: tuple = (0.4, 0.4, 0.8),
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    dry_run: bool = False,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    print("=" * 60)
    print("Building nifti index ...")
    nifti_map = build_dicom_map(nifti_root)
    print(f"  Found {len(nifti_map)} nifti files")

    # Summary across all folds
    all_stats = []

    for fold in range(5):
        print(f"\n{'=' * 60}") 
        print(f"Fold {fold}")
        print("-" * 60)

        pred_csv = data_dir / f"Prediction_extract_fold_{fold}.csv"
        df = pd.read_csv(pred_csv)

        tp = df[df["confidence_level"] == "GT"].copy()
        fp = df[df["confidence_level"].isin(["高置信度", "中置信度", "低置信度"])].copy()

        # ---- Gray-zone filter (3D IoU) ----
        # Build TP bbox index per patient_key
        tp_boxes = {}
        for _, tp_row in tp.iterrows():
            pk = tp_row["patient_key"]
            tp_boxes.setdefault(pk, []).append({
                "min": np.array([tp_row["roi_patientPos_min_x"],
                                 tp_row["roi_patientPos_min_y"],
                                 tp_row["roi_patientPos_min_z"]]),
                "max": np.array([tp_row["roi_patientPos_max_x"],
                                 tp_row["roi_patientPos_max_y"],
                                 tp_row["roi_patientPos_max_z"]]),
            })

        fp_gray_indices = []
        for idx, fp_row in fp.iterrows():
            pk = fp_row["patient_key"]
            if pk not in tp_boxes:
                continue
            fp_min = np.array([fp_row["roi_patientPos_min_x"],
                               fp_row["roi_patientPos_min_y"],
                               fp_row["roi_patientPos_min_z"]])
            fp_max = np.array([fp_row["roi_patientPos_max_x"],
                               fp_row["roi_patientPos_max_y"],
                               fp_row["roi_patientPos_max_z"]])
            for tp_box in tp_boxes[pk]:
                if compute_3d_iou(fp_min, fp_max, tp_box["min"], tp_box["max"]) > 0:
                    fp_gray_indices.append(idx)
                    break

        fp_gray = fp.loc[fp_gray_indices]
        fp_clean = fp.drop(fp_gray_indices)
        print(f"  TP: {len(tp)}")
        print(f"  FP total: {len(fp)}, clean (3D IoU==0): {len(fp_clean)}, gray (3D IoU>0): {len(fp_gray)}")

        # ---- Per-scan FP sampling (cap + 60/30/10 ratio) ----
        fp_scan_groups = fp_clean.groupby("patient_key")

        n_high_cap = int(per_scan_cap * 0.6)
        n_mid_cap = int(per_scan_cap * 0.3)
        n_low_cap = per_scan_cap - n_high_cap - n_mid_cap

        sampled_parts = []
        for scan_id, scan_fps in fp_scan_groups:
            scan_sampled = []
            for level, n_sample in [("高置信度", n_high_cap), ("中置信度", n_mid_cap), ("低置信度", n_low_cap)]:
                pool = scan_fps[scan_fps["confidence_level"] == level]
                if len(pool) == 0:
                    continue
                sampled = pool.sort_values("detector_score", ascending=False).head(n_sample)
                scan_sampled.append(sampled)
            if scan_sampled:
                sampled_parts.append(pd.concat(scan_sampled, ignore_index=True))

        if sampled_parts:
            fp_per_scan = pd.concat(sampled_parts, ignore_index=True)
        else:
            fp_per_scan = pd.DataFrame(columns=fp_clean.columns)

        n_fp_target = int(len(tp) / gt_fp_ratio)
        print(f"  Per-scan cap: {per_scan_cap} (高{n_high_cap}/中{n_mid_cap}/低{n_low_cap}), "
              f"scans with FPs: {len(fp_scan_groups)}")
        print(f"  FP after per-scan cap: {len(fp_per_scan)}, fold target: {n_fp_target} "
              f"(GT:FP = 1:{int(1/gt_fp_ratio)})")

        if len(fp_per_scan) <= n_fp_target:
            fp_sampled = fp_per_scan
            if len(fp_per_scan) < n_fp_target:
                print(f"  [WARNING] Per-scan cap limits FP to {len(fp_per_scan)} (< target {n_fp_target})")
        else:
            # Subsample to fold target, maintaining confidence ratio
            sampled_parts = []
            for level, ratio in [("高置信度", 0.6), ("中置信度", 0.3), ("低置信度", 0.1)]:
                pool = fp_per_scan[fp_per_scan["confidence_level"] == level]
                n_sample = min(int(n_fp_target * ratio), len(pool))
                if n_sample > 0:
                    sampled = pool.sort_values("detector_score", ascending=False).head(n_sample)
                    sampled_parts.append(sampled)
            fp_sampled = pd.concat(sampled_parts, ignore_index=True)
            actual_n = len(fp_sampled)
            if actual_n < n_fp_target:
                used_idx = set(fp_sampled.index)
                remaining = fp_per_scan[~fp_per_scan.index.isin(used_idx)]
                remaining = remaining.sort_values("detector_score", ascending=False)
                fill_n = min(n_fp_target - actual_n, len(remaining))
                fill = remaining.head(fill_n)
                fp_sampled = pd.concat([fp_sampled, fill], ignore_index=True)
                print(f"  Shortfall filled: +{len(fill)} from remaining pool")

        print(f"  FP sampled: {len(fp_sampled)}")

        # ---- Group by nifti path ----
        all_candidates = []

        for _, row in tp.iterrows():
            pk = row["patient_key"]
            nii_path = nifti_map.get(pk)
            if nii_path is None:
                print(f"  [SKIP] GT patient_key={pk}: no nifti found")
                continue
            all_candidates.append({
                "nii_path": nii_path,
                "patient_key": pk,
                "StudyInstanceUID": row["StudyInstanceUID"],
                "SeriesInstanceUID": row["SeriesInstanceUID"],
                "label": 1,  # TP
                "center_world": (row["roi_patientPos_center_x"],
                                 row["roi_patientPos_center_y"],
                                 row["roi_patientPos_center_z"]),
                "prediction_type": "TP",
                "confidence_level": "GT",
            })

        for _, row in fp_sampled.iterrows():
            pk = row["patient_key"]
            nii_path = nifti_map.get(pk)
            if nii_path is None:
                print(f"  [SKIP] FP patient_key={pk}: no nifti found")
                continue
            all_candidates.append({
                "nii_path": nii_path,
                "patient_key": pk,
                "StudyInstanceUID": row["StudyInstanceUID"],
                "SeriesInstanceUID": row["SeriesInstanceUID"],
                "label": 0,  # FP
                "center_world": (row["roi_patientPos_center_x"],
                                 row["roi_patientPos_center_y"],
                                 row["roi_patientPos_center_z"]),
                "prediction_type": "FP",
                "confidence_level": row["confidence_level"],
            })

        cand_df = pd.DataFrame(all_candidates)
        # Group by nii_path for efficient CT loading
        groups = cand_df.groupby("nii_path")

        print(f"  Candidates: {len(cand_df)} across {len(groups)} CT volumes")

        # ---- Extract patches ----
        fold_output_dir = output_dir / f"fold_{fold}"
        if not dry_run:
            fold_output_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = []
        patch_idx = 0
        pbar = tqdm(groups, desc=f"  Extracting patches", unit="vol")

        for nii_path, group in pbar:

            # Load CT from nifti
            ct_data, affine = load_dicom_series(nii_path)
            # Original spacing
            orig_sp = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))  # [si, sj, sk]

            # Resampling
            zoom_factors = compute_zoom(orig_sp, target_spacing)  # [zi, zj, zk]
            resampled_shape = tuple(int(s * z) for s, z in zip(ct_data.shape, zoom_factors))

            if dry_run:
                ct_resampled = np.zeros(resampled_shape, dtype=np.float32)
            else:
                ct_resampled = ndzoom(ct_data, zoom_factors, order=1, mode="nearest")

            for _, row in group.iterrows():
                center_world = np.array(row["center_world"])
                # World → original voxel
                orig_vox = world_to_orig_voxel(center_world, affine)  # [i, j, k]

                # Original voxel → resampled voxel
                resampled_vox = orig_vox * zoom_factors

                cur_jitter_range = jitter_range if row["label"] == 1 else 0

                n_patches_for_this = n_jitter if row["label"] == 1 else 1

                for ji in range(n_patches_for_this):
                    patch, meta = extract_patch(
                        ct_resampled, resampled_vox, patch_size, jitter_range=cur_jitter_range
                    )
                    if patch is None: 
                        continue

                    # # HU窗口化 + 归一化到0-255（匹配第一阶段: -1000~400）
                    if not dry_run:
                        np.clip(patch, hu_min, hu_max, out=patch)
                        patch = ((patch - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)

                        fname = f"{patch_idx:06d}_{row['patient_key']}_"
                        fname += f"{'TP' if row['label'] == 1 else 'FP'}"
                        if row["label"] == 1:
                            fname += f"_j{ji}"
                        fname += ".npy"
                        np.save(fold_output_dir / fname, patch)

                    metadata_rows.append({
                        "fold": fold,
                        "patch_idx": patch_idx,
                        "patient_key": row["patient_key"],
                        "StudyInstanceUID": row["StudyInstanceUID"],
                        "SeriesInstanceUID": row["SeriesInstanceUID"],
                        "label": row["label"],
                        "prediction_type": row["prediction_type"],
                        "confidence_level": row["confidence_level"],
                        "jitter_index": ji,
                        "center_world_x": center_world[0],
                        "center_world_y": center_world[1],
                        "center_world_z": center_world[2],
                        "patch_file": fname if not dry_run else "",
                        "checkpoint ID":f"/workspace/Train_result/Mucus_249_neg_0/Train_fold{fold}/weights/best.pt"
                    })
                    patch_idx += 1

        # Save metadata
        meta_df = pd.DataFrame(metadata_rows)
        if not dry_run:
            meta_df.to_csv(fold_output_dir / "metadata.csv", index=False)

        tp_patches = sum(1 for r in metadata_rows if r["label"] == 1)
        fp_patches = sum(1 for r in metadata_rows if r["label"] == 0)
        all_stats.append({
            "fold": fold,
            "tp_candidates": len(tp),
            "fp_total": len(fp),
            "fp_gray_removed": len(fp_gray),
            "fp_sampled": len(fp_sampled),
            "tp_patches": tp_patches,
            "fp_patches": fp_patches,
            "ratio": f"1:{fp_patches/tp_patches:.2f}" if tp_patches > 0 else "N/A",
        })
        print(f"  TP patches: {tp_patches}, FP patches: {fp_patches}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    for s in all_stats:
        print(f"  Fold {s['fold']}: TP={s['tp_patches']}, FP={s['fp_patches']}, "
              f"ratio={s['ratio']}, gray removed={s['fp_gray_removed']}")

    total_tp = sum(s["tp_patches"] for s in all_stats)
    total_fp = sum(s["fp_patches"] for s in all_stats)
    print(f"  Total: TP={total_tp}, FP={total_fp}, ratio=1:{total_fp/total_tp:.2f}" if total_tp > 0 else "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3D CT patches for 2nd-stage classification")
    parser.add_argument("--data_dir", default="/workspace/Get_3D_Class_Data/分类数据集所需数据表-260520") # 数据CSV路径，包含Prediction_TP_FP_fold_x.csv文件
    parser.add_argument("--nifti_root", default="/data/nifti_files") # nifti数据根目录
    parser.add_argument("--output_dir", default="/data/Class_3D_patch/jitter_4_260520")  # 3D patch输出路径
    parser.add_argument("--gt_fp_ratio", type=float, default=0.5,
                        help="GT:FP ratio (0.5 = 1:2, 1.0 = 1:1)")  # 正负样本的比例，0.5表示GT:FP=1:2，1.0表示GT:FP=1:1
    parser.add_argument("--per_scan_cap", type=int, default=100,
                        help="Max FPs per scan (hard-negative mining cap)")  # 每个scan最多取多少个FP，用于难负样本挖掘的上限
    parser.add_argument("--n_jitter", type=int, default=3,
                        help="Number of jittered copies per GT") # 对于每个GT样本，生成多少个带随机偏移的增强样本。FP不进行jitter。
    parser.add_argument("--jitter_range", type=float, default=4.0,
                        help="Max jitter offset in resampled voxels (±4 ≈ ±1.6mm in x/y, ±3.2mm in z)") # 每个GT样本的随机偏移范围，单位是重采样后的体素。默认±4个体素，约等于±1.6mm在x/y轴，±3.2mm在z轴。
    parser.add_argument("--patch_size", nargs=3, type=int, default=[64, 64, 32]) # 输出patch的尺寸，单位是重采样后的体素。默认64x64x32。
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.4, 0.4, 0.8]) # 重采样的目标空间分辨率，单位是mm。默认0.4mm x 0.4mm x 0.8mm。
    parser.add_argument("--hu_min", type=float, default=-1000.0) # HU窗口的最小值，默认-1000（空气）。提取的patch会被裁剪到这个范围，并线性归一化到0-255。
    parser.add_argument("--hu_max", type=float, default=400.0) # HU窗口的最大值，默认400（软组织）。提取的patch会被裁剪到这个范围，并线性归一化到0-255。
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip actual patch extraction, just simulate") # 如果设置了--dry_run，则不进行实际的patch提取和保存，只模拟流程并输出统计信息。这对于调试和验证流程非常有用。
    args = parser.parse_args()

    process_folds(
        data_dir=args.data_dir,
        nifti_root=args.nifti_root,
        output_dir=args.output_dir,
        gt_fp_ratio=args.gt_fp_ratio,
        per_scan_cap=args.per_scan_cap,
        n_jitter=args.n_jitter,
        jitter_range=args.jitter_range,
        patch_size=tuple(args.patch_size),
        target_spacing=tuple(args.target_spacing),
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        dry_run=args.dry_run,
    )
