"""Extract 3D bounding boxes from mucus plug masks.

For each mask file, use connected component analysis to find individual mucus plugs,
compute their 3D bounding boxes in both voxel and world (patient) coordinates,
and save results to a CSV. Each plug (connected component) gets its own row
with plug_id and the total plug_count for that patient.
最终还会保存每个粘液栓中心所在的CT切片图像（PNG格式），在图像上标记中心点位置，供后续检查和可视化使用。
"""
## 提取标签的3D边界框（中心点坐标和尺寸），使用连通域分析识别独立的粘液栓
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
import os
import matplotlib.pyplot as plt

# --- Config ---
CSV_PATH = "/data/yolo_dataset_249/Kfold_neg_0_backup/fold_assignment_integrated.csv"
OUT_PATH = "/workspace/Get_3D_Class_Data/分类数据集所需数据表-260520/mask_bboxes.csv"
PNG_DIR = "/workspace/all_results/temp_data/plug_slices"
WW, WL = 1500, -700  # window width, window level for CT display

# --- Load fold assignments ---
df = pd.read_csv(CSV_PATH)
print(f"Total entries: {len(df)}")

results = []
missing_masks = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing masks"):
    fold = row["fold"]
    image_path = row["path"]
    mask_path = image_path.replace("image", "mask")
    patient_key = Path(image_path).name.replace(".nii.gz", "")

    # if patient_key != "E0001179_20110603":
    #     continue  # DEBUG: only process one patient for now

    if not Path(mask_path).exists():
        missing_masks.append(mask_path)
        continue

    # Load mask
    nii = nib.load(mask_path)
    data = nii.get_fdata()
    affine = nii.affine

    # Voxel spacing (mm per voxel along each axis)
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))  # [sp_i, sp_j, sp_k]

    # Load CT image for slice export
    ct_nii = nib.load(image_path)
    ct_data = ct_nii.get_fdata().astype(np.float32)  # shape: (i, j, k)

    # Binarize and find connected components (individual mucus plugs)
    data = np.where(data > 0, 1, 0).astype(np.uint8)
    labeled_array, num_features = ndimage.label(data)

    for comp_id in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == comp_id)  # N × 3, order: [i, j, k]
        if len(coords) == 0:
            continue

        v_min = coords.min(axis=0)  # [min_i, min_j, min_k]
        v_max = coords.max(axis=0)  # [max_i, max_j, max_k]
        v_center = (v_min + v_max) / 2.0  # [ci, cj, ck]
        v_dims = v_max - v_min + 1       # [di, dj, dk]
        
        # World center: affine × [ci, cj, ck, 1]
        world_center = nib.affines.apply_affine(affine, v_center)
        # World dimensions
        world_dims = v_dims * spacing  # [di*sp_i, dj*sp_j, dk*sp_k]

        results.append({
            "patient_key": patient_key,
            "mask_path": mask_path,
            "image_path": image_path,
            "fold": int(fold),
            "plug_id": comp_id,
            "plug_count": num_features,
            # World (patient) coordinates
            "center_x": round(float(world_center[0]), 4),
            "center_y": round(float(world_center[1]), 4),
            "center_z": round(float(world_center[2]), 4),
            "width": round(float(world_dims[0]), 4),   # x
            "height": round(float(world_dims[1]), 4),   # y
            "depth": round(float(world_dims[2]), 4),   # z
            # Voxel coordinates
            "voxel_center_i": int(round(v_center[0])),
            "voxel_center_j": int(round(v_center[1])),
            "voxel_center_k": int(round(v_center[2])),
            "voxel_dim_i": int(v_dims[0]),
            "voxel_dim_j": int(v_dims[1]),
            "voxel_dim_k": int(v_dims[2]),
        })

        # Save CT slice at v_center (axial slice along k axis)
        k_slice = int(round(v_center[2]))
        if 0 <= k_slice < ct_data.shape[2]:
            slice_img = ct_data[:, :, k_slice].T  # (j, i) — rows, cols for display
            # Apply window: WW=1500, WL=-700
            lo, hi = WL - WW / 2, WL + WW / 2
            slice_img = np.clip(slice_img, lo, hi)
            slice_img = ((slice_img - lo) / (hi - lo) * 255).astype(np.uint8)
            # Save PNG with v_center marked
            png_path = os.path.join(
                PNG_DIR, patient_key,
                f"{patient_key}_plug{comp_id}_slice{k_slice}.png"
            )
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(slice_img, cmap="gray", vmin=0, vmax=255, origin="upper")
            ax.scatter(v_center[0], v_center[1], c="red", marker="+", s=100, linewidths=1.5)
            ax.axis("off")
            fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=150)
            plt.close(fig)

# --- Save ---
result_df = pd.DataFrame(results)
if not os.path.exists(os.path.dirname(OUT_PATH)):
    os.makedirs(os.path.dirname(OUT_PATH))
result_df.to_csv(OUT_PATH, index=False)

# --- Report ---
print(f"\nDone. {len(results)} bounding boxes saved to {OUT_PATH}")
print(f"Masks processed: {len(df) - len(missing_masks)}")
if missing_masks:
    print(f"Missing masks: {len(missing_masks)}")
    for m in missing_masks[:5]:
        print(f"  {m}")
    if len(missing_masks) > 5:
        print(f"  ... and {len(missing_masks) - 5} more")

# Per-fold summary
print("\nBboxes per fold:")
for f in sorted(result_df["fold"].unique()):
    count = (result_df["fold"] == f).sum()
    n_patients = result_df[result_df["fold"] == f]["patient_key"].nunique()
    print(f"  Fold {int(f)}: {count} bboxes from {n_patients} patients")
