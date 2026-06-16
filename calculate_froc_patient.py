import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
import cc3d
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist


def load_mask(mask_path):
    """
    Load a nii.gz mask file.

    Args:
        mask_path: Path to the mask file

    Returns:
        mask_array: Mask as numpy array
        info: Mask metadata dictionary
    """
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    info = {
        'origin': mask.GetOrigin(),
        'spacing': mask.GetSpacing(),
        'direction': mask.GetDirection()
    }
    return mask_array, info


def load_gt_boxes_from_csv(csv_path, fold=None):
    """
    Read GT bounding boxes (voxel coordinates) directly from a CSV,
    skipping mask loading and CC3D analysis.

    CSV columns: voxel_center_i (width/x), voxel_center_j (height/y), voxel_center_k (depth/z),
                 voxel_dim_i, voxel_dim_j, voxel_dim_k

    Returns:
        gt_boxes_dict: {patient_key: [[z_min, y_min, x_min, z_max, y_max, x_max], ...]}
    """
    df = pd.read_csv(csv_path)
    if fold is not None:
        df = df[df['fold'] == fold]

    gt_boxes_dict = defaultdict(list)
    for _, row in df.iterrows():
        patient = row['patient_key']
        ci, cj, ck = row['voxel_center_i'], row['voxel_center_j'], row['voxel_center_k']
        di, dj, dk = row['voxel_dim_i'], row['voxel_dim_j'], row['voxel_dim_k']

        x_min = ci - di / 2
        x_max = ci + di / 2
        y_min = cj - dj / 2
        y_max = cj + dj / 2
        z_min = ck - dk / 2
        z_max = ck + dk / 2

        gt_boxes_dict[patient].append([z_min, y_min, x_min, z_max, y_max, x_max])

    return gt_boxes_dict


def get_mask_metadata(mask_path):
    """Load only the metadata (origin, spacing, direction) of a mask file, without pixel data."""
    mask = sitk.ReadImage(str(mask_path))
    return {
        'origin': mask.GetOrigin(),
        'spacing': mask.GetSpacing(),
        'direction': mask.GetDirection()
    }


def mask_to_3d_boxes(mask_array):
    """
    Compute 3D bounding boxes from a 3D mask.

    Args:
        mask_array: 3D mask array (depth, height, width)

    Returns:
        boxes_3d: List of 3D bounding boxes [(z_min, y_min, x_min, z_max, y_max, x_max)]
    """
    # Binarize the mask
    binary_mask = (mask_array > 0).astype(np.uint8)

    # Connected components labeling
    labels_out = cc3d.connected_components(binary_mask, connectivity=26)  # 3D connectivity
    num_objects = labels_out.max()

    boxes_3d = []

    if num_objects == 0:
        return boxes_3d

    for obj_label in range(1, num_objects + 1):
        obj_mask = (labels_out == obj_label).astype(np.uint8)

        # Get non-zero coordinates of the object
        coords = np.where(obj_mask)
        if len(coords[0]) == 0:  # Skip if no non-zero elements found
            continue

        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()

        # Add a margin around the bounding box
        margin = 5
        z_min = max(z_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_min = max(x_min - margin, 0)
        z_max = min(z_max + margin, mask_array.shape[0] - 1)
        y_max = min(y_max + margin, mask_array.shape[1] - 1)
        x_max = min(x_max + margin, mask_array.shape[2] - 1)

        boxes_3d.append((z_min, y_min, x_min, z_max, y_max, x_max))

    return boxes_3d


def calculate_3d_iou(box1, box2):
    """
    Calculate 3D IoU for two bounding boxes.

    Args:
        box1: First 3D box [z1_min, y1_min, x1_min, z1_max, y1_max, x1_max]
        box2: Second 3D box [z2_min, y2_min, x2_min, z2_max, y2_max, x2_max]

    Returns:
        iou: 3D intersection over union
    """
    z1_min, y1_min, x1_min, z1_max, y1_max, x1_max = box1
    z2_min, y2_min, x2_min, z2_max, y2_max, x2_max = box2

    # Compute intersection bounds
    zi_min = max(z1_min, z2_min)
    yi_min = max(y1_min, y2_min)
    xi_min = max(x1_min, x2_min)
    zi_max = min(z1_max, z2_max)
    yi_max = min(y1_max, y2_max)
    xi_max = min(x1_max, x2_max)

    # Compute intersection volume
    inter_depth = max(0, zi_max - zi_min)
    inter_height = max(0, yi_max - yi_min)
    inter_width = max(0, xi_max - xi_min)
    inter_volume = inter_depth * inter_height * inter_width

    # Compute individual volumes
    vol1 = (z1_max - z1_min) * (y1_max - y1_min) * (x1_max - x1_min)
    vol2 = (z2_max - z2_min) * (y2_max - y2_min) * (x2_max - x2_min)

    # Compute union volume
    union_volume = vol1 + vol2 - inter_volume

    return inter_volume / union_volume if union_volume > 0 else 0


def calculate_froc_3d(predictions, ground_truths_dict, patient_count, iou_threshold):
    """
    Compute FROC curve using 3D bounding boxes with per-patient IoU matching.

    Args:
        predictions: List of (score, pred_box, patient_key, [optional row])
        ground_truths_dict: {patient_key: [[z_min, y_min, x_min, z_max, y_max, x_max], ...]}
        patient_count: Total number of patients
        iou_threshold: 3D IoU threshold

    Returns:
        fps: False positive rates (avg per patient)
        sensitivities: Sensitivity values
        thresholds: Corresponding confidence thresholds
    """
    # Sort predictions by confidence score
    predictions.sort(key=lambda x: x[0], reverse=True)

    # Count total ground truths
    total_gt = sum(len(boxes) for boxes in ground_truths_dict.values())
    if total_gt == 0:
        return [], [], []

    # Track which GTs have been detected per patient
    detected = {patient: [False] * len(boxes) for patient, boxes in ground_truths_dict.items()}

    # Compute TP and FP at each threshold
    tp = 0
    fp = 0
    fps = []
    sensitivities = []
    thresholds = []  # Record threshold at each point

    for pred in predictions:
        score, pred_box, patient_key = pred[0], pred[1], pred[2]

        # Match IoU only against GTs of the same patient
        max_iou = 0
        best_gt_idx = -1
        patient_gts = ground_truths_dict.get(patient_key, [])

        for i, gt_box in enumerate(patient_gts):
            if not detected[patient_key][i]:
                iou = calculate_3d_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = i

        if max_iou >= iou_threshold:
            # True positive
            tp += 1
            detected[patient_key][best_gt_idx] = True
        else:
            # False positive
            fp += 1

        # Compute sensitivity and false positive rate
        sensitivity = tp / total_gt
        avg_fp_per_patient = fp / patient_count  # Average false positives per patient

        fps.append(avg_fp_per_patient)
        sensitivities.append(sensitivity)
        thresholds.append(score)  # Confidence threshold at this point

    return fps, sensitivities, thresholds


def plot_froc_curve(fps, sensitivities, fold=0, save_path="/workspace/resutls_froc"):
    """
    Plot FROC curve.

    Args:
        fps: List of false positive rates
        sensitivities: List of sensitivity values
        fold: Fold number
        save_path: Save directory, default "/workspace/resutls_froc"
    """
    if not fps or not sensitivities:
        print("No FROC data to plot")
        return

    plt.figure(figsize=(10, 8))

    # Plot FROC curve
    plt.plot(fps, sensitivities, 'b-', linewidth=2, label='FROC Curve')

    plt.title(f'FROC Curve - Fold {fold}', fontsize=16)
    plt.xlabel('Average Number of False Positives Per Patient', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.xlim(0, max(fps) if fps else 1)
    plt.ylim(0, 1)

    plt.legend(fontsize=12)

    # Save figure
    plot_path = f"{save_path}/froc_curve_fold_{fold}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"FROC curve saved to {plot_path}")

    plt.show()


def find_sensitivity_at_target_fp(fps, sensitivities, thresholds, target_fp_per_patient):
    """
    Find the sensitivity and confidence threshold at a target FP-per-patient count.

    Args:
        fps: List of false positive rates (avg FPs per patient)
        sensitivities: List of sensitivity values
        thresholds: Corresponding confidence threshold list
        target_fp_per_patient: Target average FPs per patient

    Returns:
        sensitivity_at_target_fp: Sensitivity at the target FP count
        threshold_at_target_fp: Confidence threshold at the target FP count
    """
    if not fps:
        return 0, 0

    # Find the point closest to the target FP count
    diffs = [abs(fp - target_fp_per_patient) for fp in fps]
    closest_idx = diffs.index(min(diffs))

    sensitivity_at_target_fp = sensitivities[closest_idx]
    threshold_at_target_fp = thresholds[closest_idx] if closest_idx < len(thresholds) else 0

    return sensitivity_at_target_fp, threshold_at_target_fp


def coord_pat2vox(pat, origin, spacing, direction):
    """
    Convert world (patient) coordinates to voxel coordinates.

    Args:
        pat: World coordinates [x, y, z]
        origin: Image origin
        spacing: Voxel spacing
        direction: Direction matrix

    Returns:
        voxel_coord: Voxel coordinates (x, y, z)
    """
    origin = np.array(origin)
    spacing = np.array(spacing)
    direction = np.array(direction)
    direction_matrix = direction.reshape(3, 3)
    transformation_matrix = direction_matrix * spacing
    pat = np.array(pat)
    voxel_coord = np.linalg.inv(transformation_matrix).dot(pat - origin)
    return tuple(voxel_coord)


def match_predictions_with_labels(predictions_with_details, ground_truths_dict, patient_count, iou_threshold=0.01, min_confidence=None):
    """
    Match predictions with ground truth labels and create a new table with TP/FP labels
    (per-patient IoU matching).

    Args:
        predictions_with_details: List of (score, pred_box, patient_key, original_row_data)
        ground_truths_dict: {patient_key: [[z_min, y_min, x_min, z_max, y_max, x_max], ...]}
        patient_count: Total number of patients
        iou_threshold: 3D IoU threshold
        min_confidence: Minimum confidence threshold — predictions below this are skipped

    Returns:
        results_df: DataFrame with prediction info and TP/FP labels
    """
    if not predictions_with_details:
        return pd.DataFrame()

    # Sort predictions by confidence
    predictions_sorted = sorted(predictions_with_details, key=lambda x: x[0], reverse=True)

    # Track which GTs have been detected per patient
    detected = {patient: [False] * len(boxes) for patient, boxes in ground_truths_dict.items()}

    results = []

    for pred in predictions_sorted:
        score, pred_box, patient_key, original_row = pred

        # Skip predictions below the minimum confidence threshold
        if min_confidence is not None and score < min_confidence:
            continue

        # Match IoU only against GTs of the same patient
        max_iou = 0
        best_gt_idx = -1
        patient_gts = ground_truths_dict.get(patient_key, [])

        for i, gt_box in enumerate(patient_gts):
            if not detected[patient_key][i]:
                iou = calculate_3d_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = i

        if max_iou >= iou_threshold:
            # True positive
            label = "TP"
            detected[patient_key][best_gt_idx] = True
        else:
            # False positive
            label = "FP"

        # Convert original row data to dict
        row_dict = original_row.to_dict() if hasattr(original_row, 'to_dict') else dict(original_row)

        # Add pixel coordinate info for the predicted box
        z_min, y_min, x_min, z_max, y_max, x_max = pred_box
        row_dict['pixel_x_min'] = x_min
        row_dict['pixel_y_min'] = y_min
        row_dict['pixel_z_min'] = z_min
        row_dict['pixel_x_max'] = x_max
        row_dict['pixel_y_max'] = y_max
        row_dict['pixel_z_max'] = z_max
        row_dict['pixel_center_x'] = (x_min + x_max) / 2
        row_dict['pixel_center_y'] = (y_min + y_max) / 2
        row_dict['pixel_center_z'] = (z_min + z_max) / 2
        row_dict['pixel_width'] = x_max - x_min
        row_dict['pixel_height'] = y_max - y_min
        row_dict['pixel_depth'] = z_max - z_min

        # Add IoU value and label
        row_dict['iou'] = max_iou
        row_dict['prediction_type'] = label

        results.append(row_dict)

    results_df = pd.DataFrame(results)

    # Reorder columns: put prediction_type at the end
    if not results_df.empty:
        cols = [col for col in results_df.columns if col != 'prediction_type'] + ['prediction_type']
        results_df = results_df[cols]

    return results_df


def process_fold_predictions(fold=0, pred_csv_path=None, save_path="/workspace/all_results/results_froc", iou_threshold=0.001, min_confidence=None, gt_csv_path=None):
    """
    Process predictions for a given fold and compute FROC curve using 3D bounding boxes.

    Args:
        fold: Fold number
        pred_csv_path: Path to prediction CSV
        save_path: Output directory for results
        iou_threshold: 3D IoU threshold for TP matching
        min_confidence: Minimum confidence threshold (predictions below this are skipped)
        gt_csv_path: Path to GT_bboxes.csv — if provided, load GT boxes from CSV
                     instead of CC3D mask analysis
    """
    # Read predictions
    if not os.path.exists(pred_csv_path):
        print(f"Prediction file not found: {pred_csv_path}")
        return

    pred_df = pd.read_csv(pred_csv_path)
    print(f"Loaded {len(pred_df)} predictions")

    # If GT CSV is provided, load GT boxes directly
    gt_boxes_by_patient = None
    if gt_csv_path is not None and os.path.exists(gt_csv_path):
        print(f"Loading GT boxes from CSV: {gt_csv_path}")
        gt_boxes_by_patient = load_gt_boxes_from_csv(gt_csv_path, fold=fold)
        print(f"Loaded {sum(len(v) for v in gt_boxes_by_patient.values())} GT boxes across {len(gt_boxes_by_patient)} patients")

    # Group by patient and path
    grouped = pred_df.groupby(['patient_key', 'path'])
    print(f"Total {len(grouped)} data samples")

    all_predictions_with_details = []  # Predictions with detailed info
    all_ground_truths_dict = defaultdict(list)  # {patient_key: [3D ground truth boxes]}
    unique_patients = set()  # For counting patients
    processed_masks = set()  # Avoid re-processing the same mask
    mask_metadata_cache = {}  # Cache mask metadata

    for (patient, image_path), group in grouped:
        print(f"Processing patient: {patient}")
        print(f"Image path: {image_path}")

        unique_patients.add(patient)

        # Find mask path from image path (replace "image" with "mask")
        image_path = Path(image_path)
        if "DICOM" not in str(image_path):
            mask_dir = image_path.parent.parent

        else:
            mask_dir = image_path.parent.parent.parent

        mask_path = next(mask_dir.glob("*.nii.gz"), None)

        print(f"Mask path: {mask_path}")

        if mask_path is None or not mask_path.exists():
            print(f"Mask file not found: {mask_path}")
            continue

        # Get mask metadata (for coordinate transform), cache to avoid re-reading
        mask_path_str = str(mask_path)
        if mask_path_str not in mask_metadata_cache:
            mask_metadata_cache[mask_path_str] = get_mask_metadata(mask_path_str)
        mask_info = mask_metadata_cache[mask_path_str]

        # GT box source: prefer CSV, otherwise use mask CC3D analysis
        if gt_boxes_by_patient is not None:
            if patient in gt_boxes_by_patient and mask_path_str not in processed_masks:
                all_ground_truths_dict[patient].extend(gt_boxes_by_patient[patient])
                processed_masks.add(mask_path_str)
        else:
            # Fallback: CC3D analysis from mask
            if mask_path_str not in processed_masks:
                mask_array, mask_info = load_mask(str(mask_path))
                gt_boxes_3d = mask_to_3d_boxes(mask_array)
                all_ground_truths_dict[patient].extend(gt_boxes_3d)
                processed_masks.add(mask_path_str)

        # Collect predictions with detailed info
        for idx, row in group.iterrows():
            score = row['detector_score']

            min_x = row['roi_patientPos_min_x']
            min_y = row['roi_patientPos_min_y']
            min_z = row['roi_patientPos_min_z']
            max_x = row['roi_patientPos_max_x']
            max_y = row['roi_patientPos_max_y']
            max_z = row['roi_patientPos_max_z']

            # Convert bounding box corners to voxel coordinates
            min_vox = coord_pat2vox(
                [min_x, min_y, min_z],
                mask_info['origin'],
                mask_info['spacing'],
                mask_info['direction']
            )
            max_vox = coord_pat2vox(
                [max_x, max_y, max_z],
                mask_info['origin'],
                mask_info['spacing'],
                mask_info['direction']
            )

            x1 = min_vox[0]  # x coordinate
            y1 = min_vox[1]  # y coordinate
            z1 = min_vox[2]  # z coordinate
            x2 = max_vox[0]  # x coordinate
            y2 = max_vox[1]  # y coordinate
            z2 = max_vox[2]  # z coordinate

            # Create 3D bounding box [z_min, y_min, x_min, z_max, y_max, x_max]
            pred_box = [z1, y1, x1, z2, y2, x2]

            # Save prediction with original row data for downstream use
            all_predictions_with_details.append((score, pred_box, patient, row))

    # Compute patient count
    patient_count = len(unique_patients)
    total_gt = sum(len(v) for v in all_ground_truths_dict.values())
    print(f"Total patients: {patient_count}")
    print(f"3D Ground Truth count: {total_gt}")
    print(f"Prediction count: {len(all_predictions_with_details)}")

    # Create result table with TP/FP labels
    results_df = match_predictions_with_labels(
        all_predictions_with_details, all_ground_truths_dict, patient_count, iou_threshold, min_confidence
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Save results with TP/FP labels
    if not results_df.empty:
        results_output_path = f"{save_path}/Prediction_TP_FP_fold_{fold}.csv"
        results_df.to_csv(results_output_path, index=False)
        print(f"Results with TP/FP labels saved to {results_output_path}")
        print(f"TP count: {(results_df['prediction_type'] == 'TP').sum()}")
        print(f"FP count: {(results_df['prediction_type'] == 'FP').sum()}")

    # Compute FROC curve
    fps, sensitivities, thresholds = calculate_froc_3d(all_predictions_with_details, all_ground_truths_dict, patient_count, iou_threshold)

    # Save results
    froc_df = pd.DataFrame({
        'fps': fps,
        'sensitivity': sensitivities,
        'thresholds': thresholds
    })
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    froc_df.to_csv(f"{save_path}/froc_fold_{fold}.csv", index=False)
    print(f"FROC data saved to {save_path}/froc_fold_{fold}.csv")

    # Plot FROC curve
    plot_froc_curve(fps, sensitivities, fold, save_path)

    # Find sensitivity and confidence threshold at 100 FP/patient
    target_fp_per_patient = 100
    target_sensitivity, target_threshold = find_sensitivity_at_target_fp(fps, sensitivities, thresholds, target_fp_per_patient)

    print("FROC calculation complete")
    print(f"Max sensitivity: {max(sensitivities) if sensitivities else 0:.4f}")
    print(f"FPs at max sensitivity: {fps[sensitivities.index(max(sensitivities))] if sensitivities else 0:.4f}")
    print(f"Sensitivity at {target_fp_per_patient} FP/patient: {target_sensitivity:.4f}")
    print(f"Confidence threshold at {target_fp_per_patient} FP/patient: {target_threshold:.4f}")


if __name__ == "__main__":
    # Compute patient-level FROC curve using 3D bounding box IoU matching,
    # and save the result table with TP/FP labels
    save_path = "/workspace/all_results/results_froc/249_neg_0_test_me"
    fold = 0
    pred_csv_path = f"/workspace/all_results/predictions_by_fold/249_neg_0/Prediction_fold_{fold}.csv"
    # Process predictions for fold 0
    # Set min_confidence to filter low-confidence predictions, e.g. 0.5
    process_fold_predictions(fold=fold, pred_csv_path=pred_csv_path, save_path=save_path, iou_threshold=0.001, min_confidence=None)
