#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学影像目标检测结果可视化工具

功能：
    1. 读取包含目标检测预测结果的CSV文件
    2. 加载DICOM序列数据
    3. 将预测的物理坐标转换为体素坐标
    4. 在图像上绘制边界框和置信度分数
    5. 为每个预测生成包含多个相邻切片的合成图像
    6. 按StudyInstanceUID和SeriesInstanceUID组织目录结构并保存结果

输入：
    - CSV文件，包含以下字段：
      - path：DICOM序列路径
      - roi_patientPos_center_x, roi_patientPos_center_y, roi_patientPos_center_z：目标中心点坐标
      - diameter：目标直径
      - userAnnotComment.annotation：预测置信度

输出：
    - 按StudyInstanceUID和SeriesInstanceUID组织的目录结构
    - 每个预测点生成的JPG图像，文件名格式为{z}_{y}_{x}.jpg
    - 图像包含7个相邻切片的合成视图，每个切片上有红色边界框和置信度分数

依赖库：
    - os：目录操作
    - SimpleITK：DICOM图像处理
    - pandas：CSV文件处理
    - argparse：命令行参数解析
    - numpy：数组操作
    - cv2：图像处理和绘制
    - utils：包含坐标转换、DICOM加载、预处理和后处理函数

使用方法：
    python visualize_results.py -i <csv文件路径>
"""
import os
import SimpleITK as sitk
import pandas as pd
import argparse
import numpy as np
import cv2
from utils import coord_pat2vox, load_dicom_series, pre_processing, post_processing

def draw_boxes(pred_csv, output_dir):
    # CSV 파일 읽기
    pred_df = pd.read_csv(pred_csv)

    # 출력 폴더 준비
    os.makedirs(output_dir, exist_ok=True)

    for path in pred_df['path'].unique():
        image, info = load_dicom_series(path)
        spacing = info['spacing']
        direction = info['direction']
        origin = info['origin']

        img = pre_processing(image)
        img = np.stack([img, img, img], axis=-1)
        img_pred = img.copy()

        # Prediction 박스
        get_coords = list()
        from collections import defaultdict
        pred_boxes = pred_df[pred_df['path'] == path].copy()
        pred_eval_result = defaultdict(list)
        for _, row in pred_boxes.iterrows():
            radius_xy = int(float(row["diameter"]) / spacing[0]) // 2
            radius_z = int(float(row["diameter"]) / spacing[2]) // 2
            volume_coord = coord_pat2vox(np.array([row["roi_patientPos_center_x"], row["roi_patientPos_center_y"], row["roi_patientPos_center_z"]]), origin, spacing, direction)
            x1, y1, x2, y2 = int(volume_coord[0]) - radius_xy, int(volume_coord[1]) - radius_xy, int(volume_coord[0]) + radius_xy, int(volume_coord[1]) + radius_xy
            x = int(volume_coord[0])
            y = int(volume_coord[1])
            z = int(volume_coord[2])
            score = float(row.get("userAnnotComment.annotation", 0))
            for z in range(z, z+1):
                get_coords.append([z, y, x])
                if 0 < z-1:  
                    cv2.rectangle(img_pred[z-1,:,:,:], (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색
                    cv2.putText(img_pred[z-1,:,:,:], f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

                cv2.rectangle(img_pred[z,:,:,:], (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색
                cv2.putText(img_pred[z,:,:,:], f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1) 

                if z+1 < img_pred.shape[0]:
                    cv2.rectangle(img_pred[z+1,:,:,:], (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색
                    cv2.putText(img_pred[z+1,:,:,:], f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        if not os.path.isdir(os.path.join(output_dir, info['StudyInstanceUID'], info['SeriesInstanceUID'])):
            os.makedirs(os.path.join(output_dir, info['StudyInstanceUID'], info['SeriesInstanceUID']))

        for i, (get_z, get_y, get_x) in enumerate(get_coords):
            sub1 = get_z - 1 if get_z - 1 > 0 else 0
            sub2 = get_z - 2 if get_z - 2 > 0 else sub1
            sub3 = get_z - 3 if get_z - 3 > 0 else sub2
            plus1 = get_z + 1 if get_z + 1 < img_pred.shape[0] else get_z
            plus2 = get_z + 2 if get_z + 2 < img_pred.shape[0] else plus1
            plus3 = get_z + 3 if get_z + 3 < img_pred.shape[0] else plus2
            left_upper_y = get_y - 64 if get_y - 64 > 0 else 0
            left_upper_x = get_x - 64 if get_x - 64 > 0 else 0
            right_under_y = get_y + 64 if get_y + 64 < img_pred.shape[1] else img_pred.shape[1] - 1
            right_under_x = get_x + 64 if get_x + 64 < img_pred.shape[2] else img_pred.shape[2] - 1
            result_img = np.hstack((
                                    img_pred[sub3,left_upper_y:right_under_y,left_upper_x:right_under_x,:],
                                    img_pred[sub2,left_upper_y:right_under_y,left_upper_x:right_under_x,:],
                                    img_pred[sub1,left_upper_y:right_under_y,left_upper_x:right_under_x,:],
                                    img_pred[get_z,left_upper_y:right_under_y,left_upper_x:right_under_x:],
                                    img_pred[plus1,left_upper_y:right_under_y,left_upper_x:right_under_x,:],
                                    img_pred[plus2,left_upper_y:right_under_y,left_upper_x:right_under_x,:],
                                    img_pred[plus3,left_upper_y:right_under_y,left_upper_x:right_under_x,:],
                                    ))

            output_filename = str(get_z) + "_" + str(get_y) + "_" + str(get_x) + ".jpg"
            
            if not os.path.isdir(os.path.join(output_dir, info['StudyInstanceUID'], info['SeriesInstanceUID'])):
                os.makedirs(os.path.join(output_dir, info['StudyInstanceUID'], info['SeriesInstanceUID']))
            cv2.imwrite(os.path.join(output_dir, info['StudyInstanceUID'], info['SeriesInstanceUID'], output_filename), result_img)
        print(f"[INFO] 저장 완료: {os.path.join(output_dir, info['StudyInstanceUID'], info['SeriesInstanceUID'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction visualization")
    parser.add_argument("-i", "--csv_path", required=True, help="csv file path")
    args = parser.parse_args()

    if not os.path.isdir("./results"):
        os.makedirs("./results")

    if os.path.isfile(args.csv_path):
        csv_path = args.csv_path
    else:
        raise FileNotFoundError(f"No csv file found in the provided directory: {args.csv_path}")

    draw_boxes(csv_path, "./results")