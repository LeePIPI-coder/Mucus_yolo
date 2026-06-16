#!/usr/bin/env python3
"""基于 process.txt 加载 image_modified.nii.gz，用 YOLO 推理并输出检测结果."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import tqdm
from ultralytics import YOLO

from utils.utils import post_processing


def vox2world(vox, affine):
    """将体素坐标 [x, y, z] 转为世界坐标 (patient coordinate)."""
    vox = np.array(vox, dtype=np.float64)
    world = affine[:3, :3] @ vox + affine[:3, 3]
    return tuple(world)


def predict(nii_path, inference_model):
    patch_size = 128
    stride_hw = 64
    resize_to = 512
    depth_channel = 3
    threshold = 0.01

    new_df = defaultdict(list)

    # 加载 nii.gz
    data = nib.load(nii_path).get_fdata().astype(np.float32)
    # nibabel shape: (W, H, D)，转为原代码的 (D, H, W) 即 depth, height, width
    image = data.transpose(2, 1, 0)
    depth, height, width = image.shape

    affine = nib.load(nii_path).affine

    for ch in tqdm.tqdm(range(depth), desc=f"Inference {Path(nii_path).stem}"):
        patch_2p5d = np.zeros((height, width, depth_channel), dtype=np.float32)
        for z, re_ch in enumerate(range(ch - 1, ch + 2)):
            re_ch = np.clip(re_ch, 0, depth - 1)
            patch_2p5d[:, :, z] = image[re_ch, :, :]

        for y in range(0, height - patch_size + 1, stride_hw):
            for x in range(0, width - patch_size + 1, stride_hw):
                patch_img = patch_2p5d[y:y + patch_size, x:x + patch_size, :]
                patch_img_resized = cv2.resize(patch_img, (resize_to, resize_to),
                                               interpolation=cv2.INTER_LINEAR)

                outputs = inference_model.predict(source=patch_img_resized, save=False,
                                                  imgsz=512, verbose=False)
                if len(outputs[0].boxes) == 0:
                    continue

                boxes = outputs[0].boxes.xyxy.detach().cpu().numpy()
                scores = outputs[0].boxes.conf.detach().cpu().numpy()

                for box, score in zip(boxes, scores):
                    if score < threshold:
                        continue

                    x1_orig = int(box[0] * patch_size / resize_to) + x
                    y1_orig = int(box[1] * patch_size / resize_to) + y
                    x2_orig = int(box[2] * patch_size / resize_to) + x
                    y2_orig = int(box[3] * patch_size / resize_to) + y

                    diax = abs(x2_orig - x1_orig)
                    diay = abs(y2_orig - y1_orig)
                    diaz = max(diax, diay)

                    z1_orig = max(ch - (diaz // 2), 0)
                    z2_orig = min(ch + (diaz // 2), depth - 1)

                    center_x = (x1_orig + x2_orig) / 2
                    center_y = (y1_orig + y2_orig) / 2
                    center_z = float(ch)

                    world_center = vox2world([center_x, center_y, center_z], affine)
                    world_min = vox2world([x1_orig, y1_orig, z1_orig], affine)
                    world_max = vox2world([x2_orig, y2_orig, z2_orig], affine)

                    new_df['StudyInstanceUID'].append(Path(nii_path).parent.name)
                    new_df['SeriesInstanceUID'].append('nifti')
                    new_df['roi_patientPos_min_x'].append(world_min[0])
                    new_df['roi_patientPos_min_y'].append(world_min[1])
                    new_df['roi_patientPos_min_z'].append(world_min[2])
                    new_df['roi_patientPos_max_x'].append(world_max[0])
                    new_df['roi_patientPos_max_y'].append(world_max[1])
                    new_df['roi_patientPos_max_z'].append(world_max[2])
                    new_df['roi_patientPos_center_x'].append(world_center[0])
                    new_df['roi_patientPos_center_y'].append(world_center[1])
                    new_df['roi_patientPos_center_z'].append(world_center[2])
                    new_df['roi_patient_diameter_x'].append(abs(world_max[0] - world_min[0]))
                    new_df['roi_patient_diameter_y'].append(abs(world_max[1] - world_min[1]))
                    new_df['roi_patient_diameter_z'].append(abs(world_max[2] - world_min[2]))
                    new_df['LesionType'].append('ELesionAnnotType_ROI_3D')
                    new_df['detector_score'].append(score)
                    new_df['x'].append(world_center[0])
                    new_df['y'].append(world_center[1])
                    new_df['z'].append(world_center[2])
                    new_df['diameter'].append(max(diax, diay, diaz))
                    new_df['diameter_x'].append(diax)
                    new_df['diameter_y'].append(diay)
                    new_df['diameter_z'].append(diaz)
                    new_df['path'].append(str(nii_path))

    new_df = pd.DataFrame(new_df)
    if len(new_df) == 0:
        return new_df
    filtered_df = post_processing(new_df)
    return filtered_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_pth",
                        default='/workspace/Train_result/Mucus_249_neg_0/Train_fold0/weights/best.pt',
                        help="YOLO model weight path")
    parser.add_argument("-t", "--txt",
                        default='/workspace/process.txt',
                        help="process.txt path with subdirectory names")
    parser.add_argument("-r", "--root",
                        default='/data/Mucus_data',
                        help="Root directory containing subdirectories")
    parser.add_argument("-o", "--output",
                        default='./results_niigz.csv',
                        help="Output CSV path")
    args = parser.parse_args()

    # 读取 process.txt 中的名称
    root_dir = Path(args.root)
    with open(args.txt) as f:
        names = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            names.append(re.split(r'\s+', line)[-1])

    inference_model = YOLO(args.model_pth)
    root_df = defaultdict(list)

    for name in names:
        nii_path = root_dir / name / "image.nii_modified.nii.gz"
        if not nii_path.exists():
            print(f"[SKIP] {name}: file not found: {nii_path}")
            continue
        print(f"[INFO] Processing: {nii_path}")
        filtered_df = predict(str(nii_path), inference_model)
        for key, value in filtered_df.items():
            root_df[key].extend(value)

    root_df = pd.DataFrame(root_df)
    root_df.to_csv(args.output, index=False)
    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
