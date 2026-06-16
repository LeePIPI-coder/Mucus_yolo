"""Inference + FROC evaluation for 2nd-stage FP reduction classifier.

Workflow per fold:
  1. Load Prediction_TP_FP_fold_x.csv (full 1st-stage candidates)
  2. Extract 3D patches for all candidates (grouped by nifti, resample once per CT)
  3. Run batch inference → classifier_score per candidate
  4. Save augmented CSV with classifier_score column
  5. Re-rank by classifier_score, match against GT masks (3D IoU), compute FROC
  6. Compare 2nd-stage FROC vs 1st-stage baseline (Sens@1FP, Sens@2FP, etc.)
"""

import argparse
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ---- Reuse extraction utilities from data preparation ----
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Get_3D_Class_Data"))
from extract_patches import (
    build_dicom_map,
    load_dicom_series,
    world_to_orig_voxel,
    compute_zoom,
)

# ---- Reuse FROC utilities from 1st-stage evaluation ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calculate_froc import (
    load_mask,
    mask_to_3d_boxes,
    calculate_3d_iou,
)

from model import build_fp_classifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET_SPACING = (0.4, 0.4, 0.8)
PATCH_SIZE = (64, 64, 32)
HU_MIN, HU_MAX = -1000.0, 400.0


def build_mask_map(nifti_root):
    """Map patient_key → mask nii.gz path (from *mask* subdirectories)."""
    root = Path(nifti_root)
    mask_map = {}
    for nii_path in root.rglob("*.nii.gz"):
        if "mask" not in nii_path.parent.name:
            continue
        patient_key = nii_path.name.replace(".nii.gz", "")
        mask_map[patient_key] = str(nii_path)
    return mask_map


def extract_one_patch(ct_resampled, center_ijk, patch_size=PATCH_SIZE):
    """Crop a single patch around center_ijk (no jitter). Returns patch or None."""
    pi, pj, pk = patch_size
    ri, rj, rk = pi // 2, pj // 2, pk // 2

    ci, cj, ck = center_ijk
    i0 = int(round(ci)) - ri
    i1 = i0 + pi
    j0 = int(round(cj)) - rj
    j1 = j0 + pj
    k0 = int(round(ck)) - rk
    k1 = k0 + pk

    shape = ct_resampled.shape
    if i0 < 0 or j0 < 0 or k0 < 0 or i1 > shape[0] or j1 > shape[1] or k1 > shape[2]:
        return None

    patch = ct_resampled[i0:i1, j0:j1, k0:k1].copy()
    np.clip(patch, HU_MIN, HU_MAX, out=patch)
    patch = ((patch - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    return patch


class CandidateInferenceDataset(Dataset):
    """Lightweight Dataset for batch inference: loads pre-extracted patches into tensors."""

    def __init__(self, patches):
        self.patches = patches  # list of (npy_array, idx)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, orig_idx = self.patches[idx]
        x = patch.astype(np.float32) / 255.0
        x = x[np.newaxis, ...]  # (64,64,32) → (1,64,64,32)
        return torch.from_numpy(x), orig_idx


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, candidates, nifti_map, device, batch_size=32):
    """Extract patches for all candidates and run batch inference.

    Parameters
    ----------
    model : FPClassifier
    candidates : list[dict]
        Each dict has: patient_key, center_world_x/y/z, plus original row index.
    nifti_map : dict
        patient_key → nii_path
    device : torch.device
    batch_size : int

    Returns
    -------
    scores : np.ndarray  — classifier probabilities, aligned with candidates.
    """
    model.eval()

    # Group candidates by nifti_path
    groups = defaultdict(list)
    for idx, cand in enumerate(candidates):
        nii_path = nifti_map.get(cand["patient_key"])
        if nii_path is None:
            continue
        groups[nii_path].append((idx, cand))

    all_scores = np.full(len(candidates), np.nan, dtype=np.float32)

    pbar = tqdm(groups.items(), desc="  Extracting & inferring", unit="vol")
    
    for nii_path, group in pbar:
        
        ct_data, affine = load_dicom_series(nii_path)
        orig_sp = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        zoom = compute_zoom(orig_sp, TARGET_SPACING)
        from scipy.ndimage import zoom as ndzoom
        ct_resampled = ndzoom(ct_data, zoom, order=1, mode="nearest")

        patches = []
        for orig_idx, cand in group:
            center_world = np.array([cand["center_world_x"],
                                     cand["center_world_y"],
                                     cand["center_world_z"]])
            orig_vox = world_to_orig_voxel(center_world, affine)
            resampled_vox = orig_vox * zoom
            patch = extract_one_patch(ct_resampled, resampled_vox)
            if patch is not None:
                patches.append((patch, orig_idx))
            else:
                all_scores[orig_idx] = 0.0

        if not patches:
            continue

        ds = CandidateInferenceDataset(patches)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

        for batch_patches, batch_indices in loader:
            batch_patches = batch_patches.to(device)
            with autocast(device.type):
                logits = model(batch_patches)
                preds = torch.sigmoid(logits)
            for i, orig_idx in enumerate(batch_indices):
                all_scores[orig_idx.item()] = preds[i].item()

    # Fill any remaining NaN with 0
    all_scores = np.nan_to_num(all_scores, nan=0.0)

    return all_scores


def evaluate_fold(model, fold, output_dir, nifti_root, data_dir, device, batch_size=32):
    """Complete evaluation pipeline for one fold.

    Returns dict with FROC metrics at standard FP/scan points.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load prediction CSV ----
    pred_csv = Path(data_dir) / f"Prediction_fold_{fold}.csv"
    df = pd.read_csv(pred_csv)
    print(f"Fold {fold}: {len(df)} candidates loaded")

    # ---- 2. Build nifti map ----
    nifti_map = build_dicom_map(nifti_root)
    mask_map = build_mask_map(nifti_root)

    # ---- 3. Extract patches + run inference ----
    candidates = []
    for _, row in df.iterrows():
        candidates.append({
            "patient_key": str(row["patient_key"]),
            "center_world_x": float(row["roi_patientPos_center_x"]),
            "center_world_y": float(row["roi_patientPos_center_y"]),
            "center_world_z": float(row["roi_patientPos_center_z"]),
        })

    scores = run_inference(model, candidates, nifti_map, device, batch_size)

    # ---- 4. Save augmented CSV ----
    df["classifier_score"] = scores
    df.to_csv(output_dir / f"Prediction_TP_FP_fold_{fold}_with_classifier.csv", index=False)

    # ---- 5. 1st-stage FROC (baseline) ----
    baseline_metrics, baseline_total_sens = compute_froc_metrics(
        df, "detector_score", mask_map, fold)

    # ---- 6. 2nd-stage FROC (classifier re-rank) ----
    classifier_metrics, classifier_total_sens = compute_froc_metrics(
        df, "classifier_score", mask_map, fold)

    # ---- 7. Report ----
    fp_points = [1, 2]
    print(f"\n{'='*70}")
    print(f"Fold {fold} FROC Results")
    print(f"{'='*70}")
    header = f"{'FP/scan':>8}  {'1st-Stage':>12}  {'2nd-Stage':>12}  {'Delta':>10}"
    print(header)
    print("-" * 70)
    for fp in fp_points:
        s1 = baseline_metrics.get(fp, 0)
        s2 = classifier_metrics.get(fp, 0)
        delta = s2 - s1
        print(f"{fp:>8.3f}  {s1:>12.4f}  {s2:>12.4f}  {delta:>+10.4f}")
    print("-" * 70)
    print(f"{'Total':>8}  {baseline_total_sens:>12.4f}  {classifier_total_sens:>12.4f}")

    # ---- 8. Save FROC CSV ----
    froc_rows = []
    for fp in fp_points:
        froc_rows.append({
            "fp_per_scan": fp,
            "sensitivity_1st_stage": baseline_metrics.get(fp, 0),
            "sensitivity_2nd_stage": classifier_metrics.get(fp, 0),
        })
    froc_rows.append({
        "fp_per_scan": "total",
        "sensitivity_1st_stage": baseline_total_sens,
        "sensitivity_2nd_stage": classifier_total_sens,
    })
    pd.DataFrame(froc_rows).to_csv(output_dir / f"froc_comparison_fold_{fold}.csv", index=False)

    return froc_rows


def _world_to_mask_pixel(x, y, z, mask_image):
    """Convert world/LPS coordinates (mm) to mask array pixel indices [pz, py, px]."""
    direction = np.array(mask_image.GetDirection()).reshape(3, 3)
    spacing = np.array(mask_image.GetSpacing())
    origin = np.array(mask_image.GetOrigin())
    M = direction @ np.diag(spacing)
    M_inv = np.linalg.inv(M)
    idx = M_inv @ (np.array([x, y, z]) - origin)
    return idx[2], idx[1], idx[0]  # [pz, py, px] for mask array indexing


def compute_froc_metrics(df, score_col, mask_map, fold):
    """Compute FROC metrics at standard FP/scan points for a given score column.

    Per-patient GT matching: each prediction is matched only against the GT
    boxes of the same patient (not cross-patient). This prevents false TPs
    from a prediction accidentally overlapping another patient's GT.

    Parameters
    ----------
    df : DataFrame containing prediction rows
    score_col : str — column to rank by (e.g. 'detector_score' or 'classifier_score')
    mask_map : dict — patient_key → mask nii.gz path
    fold : int

    Returns
    -------
    dict — {fp_per_scan: sensitivity}
    """
    has_pixel_cols = "pixel_z_min" in df.columns
    iou_threshold = 0.0001

    # Collect all predictions and GTs globally, then match across all patients
    all_preds = []  # list of (score, pred_box)
    all_gt_boxes = []  # list of gt_box
    unique_patients = set()
    processed_masks = {}  # mask_path → (gt_boxes, mask_img_or_None)

    patient_groups = df.groupby("patient_key")

    for patient_key, group in patient_groups:
        mask_path = mask_map.get(str(patient_key))
        if mask_path is None:
            continue
        unique_patients.add(str(patient_key))

        # ---- Load GT boxes and mask once per patient ----
        if mask_path not in processed_masks:
            mask_array, _ = load_mask(mask_path)
            gt_boxes = mask_to_3d_boxes(mask_array)
            mask_img = None if has_pixel_cols else sitk.ReadImage(mask_path)
            processed_masks[mask_path] = (gt_boxes, mask_img)
        else:
            gt_boxes, mask_img = processed_masks[mask_path]

        all_gt_boxes.extend(gt_boxes)

        # ---- Convert this patient's predictions to (score, 3d_box) ----
        for _, row in group.iterrows():
            score = float(row[score_col])
            if has_pixel_cols:
                bbox = [
                    float(row["pixel_z_min"]),
                    float(row["pixel_y_min"]),
                    float(row["pixel_x_min"]),
                    float(row["pixel_z_max"]),
                    float(row["pixel_y_max"]),
                    float(row["pixel_x_max"]),
                ]
            else:
                pz_min, py_min, px_min = _world_to_mask_pixel(
                    float(row["roi_patientPos_min_x"]),
                    float(row["roi_patientPos_min_y"]),
                    float(row["roi_patientPos_min_z"]),
                    mask_img,
                )
                pz_max, py_max, px_max = _world_to_mask_pixel(
                    float(row["roi_patientPos_max_x"]),
                    float(row["roi_patientPos_max_y"]),
                    float(row["roi_patientPos_max_z"]),
                    mask_img,
                )
                bbox = [pz_min, py_min, px_min, pz_max, py_max, px_max]
            all_preds.append((score, bbox))

    if len(all_preds) == 0 or len(all_gt_boxes) == 0:
        return {}, 0.0

    # ---- Global matching: sort all predictions by score, match against all GTs ----
    all_preds.sort(key=lambda x: x[0], reverse=True)
    detected = [False] * len(all_gt_boxes)
    total_gt_count = len(all_gt_boxes)

    scored_pairs = []
    for score, pred_box in all_preds:
        max_iou = 0
        best_gt_idx = -1
        for idx, gt_box in enumerate(all_gt_boxes):
            if not detected[idx]:
                iou = calculate_3d_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = idx

        if max_iou >= iou_threshold:
            scored_pairs.append((score, True))
            detected[best_gt_idx] = True
        else:
            scored_pairs.append((score, False))

    # ---- Global FROC: sort all scored pairs by score desc ----
    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    fps = []
    sensitivities = []

    for _, is_tp in scored_pairs:
        if is_tp:
            tp += 1
        else:
            fp += 1
        fps.append(fp / len(unique_patients))
        sensitivities.append(tp / total_gt_count)

    total_sensitivity = tp / total_gt_count

    # ---- Interpolate sensitivity at target FP/scan points ----
    if len(fps) < 2:
        return {}, total_sensitivity

    fp_points = [1/8, 1/4, 1/2, 1, 2, 4, 8]
    from scipy.interpolate import interp1d
    fps_arr = np.array(fps)
    sens_arr = np.array(sensitivities)
    increasing = np.concatenate([[True], fps_arr[1:] > fps_arr[:-1]])
    fps_arr = fps_arr[increasing]
    sens_arr = sens_arr[increasing]

    if len(fps_arr) < 2:
        return {}, total_sensitivity

    f_interp = interp1d(fps_arr, sens_arr, kind="linear",
                        bounds_error=False, fill_value=(0, sens_arr[-1]))

    metrics = {}
    for fp in fp_points:
        metrics[fp] = float(f_interp(fp))

    return metrics, total_sensitivity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 2nd-stage classifier with FROC")
    parser.add_argument("--patch_root", default="/workspace/Get_3D_Class_Data/一阶段预测数据") # 跟extract_patches.py的data_dir一致
    parser.add_argument("--nifti_root", default="/data/nifti_sfiles")  # nifti数据根目录
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold to evaluate (0-4)")
    parser.add_argument("--model_path", default="/workspace/3D_resnet_class/train_results/A/re34_pretrain_neg_1/fold_0/best_model.pth",
                        help="Path to trained best_model.pth")
    parser.add_argument("--backbone", default="resnet34",
                        choices=["resnet18", "resnet34"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="/workspace/3D_resnet_class/eval_results/re34_pretrain_neg_1_260604")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.model_path, map_location=device)
    model = build_fp_classifier(args.backbone)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Model loaded from {args.model_path} "
          f"(epoch={ckpt.get('epoch')})")

    _ = evaluate_fold(
        model=model,
        fold=args.fold, 
        output_dir=args.output_dir,
        nifti_root=args.nifti_root,
        data_dir=args.patch_root, # 该目录下有Prediction_TP_FP_fold_{fold}.csv文件
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
