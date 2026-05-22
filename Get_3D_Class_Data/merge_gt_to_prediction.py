"""Append GT bounding boxes from mask_bboxes.csv into Prediction_FP_fold_x.csv files,
matched by fold number.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/workspace/Get_3D_Class_Data/分类数据集所需数据表-260520")
PRE_DIR = Path("/workspace/Get_3D_Class_Data/一阶段预测数据") 
gt = pd.read_csv(DATA_DIR / "mask_bboxes.csv")

for fold in sorted(gt["fold"].unique()):
    fold = int(fold)
    pred_file = PRE_DIR / f"Prediction_fold_{fold}.csv"
    pred = pd.read_csv(pred_file)
    gt_fold = gt[gt["fold"] == fold].copy()

    rows = []
    for _, g in gt_fold.iterrows():
        # Compute pixel min/max from voxel centers and dims
        # voxel: i=row(y), j=col(x), k=slice(z)
        px_min = g["voxel_center_j"] - g["voxel_dim_j"] / 2
        py_min = g["voxel_center_i"] - g["voxel_dim_i"] / 2
        pz_min = g["voxel_center_k"] - g["voxel_dim_k"] / 2
        px_max = g["voxel_center_j"] + g["voxel_dim_j"] / 2
        py_max = g["voxel_center_i"] + g["voxel_dim_i"] / 2
        pz_max = g["voxel_center_k"] + g["voxel_dim_k"] / 2

        # World min/max
        wx_min = g["center_x"] - g["width"] / 2
        wy_min = g["center_y"] - g["height"] / 2
        wz_min = g["center_z"] - g["depth"] / 2
        wx_max = g["center_x"] + g["width"] / 2
        wy_max = g["center_y"] + g["height"] / 2
        wz_max = g["center_z"] + g["depth"] / 2

        rows.append({
            "StudyInstanceUID": "",
            "SeriesInstanceUID": "",
            "roi_patientPos_min_x": round(wx_min, 6),
            "roi_patientPos_min_y": round(wy_min, 6),
            "roi_patientPos_min_z": round(wz_min, 6),
            "roi_patientPos_max_x": round(wx_max, 6),
            "roi_patientPos_max_y": round(wy_max, 6),
            "roi_patientPos_max_z": round(wz_max, 6),
            "roi_patientPos_center_x": g["center_x"],
            "roi_patientPos_center_y": g["center_y"],
            "roi_patientPos_center_z": g["center_z"],
            "roi_patient_diameter_x": g["width"],
            "roi_patient_diameter_y": g["height"],
            "roi_patient_diameter_z": g["depth"],
            "LesionType": "ELesionAnnotType_ROI_3D",
            "userAnnotComment.annotation": 1.0,
            "x": g["center_x"],
            "y": g["center_y"],
            "z": g["center_z"],
            "diameter": round(float(np.mean([g["width"], g["height"], g["depth"]])), 4),
            "diameter_x": g["width"],
            "diameter_y": g["height"],
            "diameter_z": g["depth"],
            "path": g["image_path"],
            "patient_key": g["patient_key"],
            "val_fold": fold,
            "pixel_x_min": px_min,
            "pixel_y_min": py_min,
            "pixel_z_min": pz_min,
            "pixel_x_max": px_max,
            "pixel_y_max": py_max,
            "pixel_z_max": pz_max,
            "pixel_center_x": g["voxel_center_j"],
            "pixel_center_y": g["voxel_center_i"],
            "pixel_center_z": g["voxel_center_k"],
            "pixel_width": g["voxel_dim_j"],
            "pixel_height": g["voxel_dim_i"],
            "pixel_depth": g["voxel_dim_k"],
            "iou": 1.0,
            "prediction_type": "TP",
            "confidence_level": "GT",
        })

    gt_df = pd.DataFrame(rows)
    merged = pd.concat([pred, gt_df], ignore_index=True)

    tp_count = (merged["prediction_type"] == "TP").sum()
    fp_count = (merged["prediction_type"] == "FP").sum()
    Extract_file = DATA_DIR / f"Prediction_extract_fold_{fold}.csv"
    merged.to_csv(Extract_file, index=False)
    print(f"Fold {fold}: {len(pred)} FP + {len(gt_df)} GT = {len(merged)} rows (TP={tp_count}, FP={fp_count})")
