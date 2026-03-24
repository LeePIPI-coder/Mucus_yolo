#!/usr/bin/env python3
"""
根据CSV文件读取指定折的CT数据，进行裁剪、模型预测并计算FROC指标
该脚本结合了make_patch_dataset.py的裁剪功能和caculate_FROC_3d.py的评估功能
使用YOLO模型进行预测
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import tempfile
import torch
import torch.nn as nn
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm import tqdm

from data_code.logging import get_logger
import matplotlib
# 设置matplotlib支持中文字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_model(model_path: str, logger):
    """
    加载YOLO模型
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        logger.info(f"YOLO模型加载成功: {model_path}")
        return model
    except ImportError:
        logger.error("错误：未安装ultralytics库，请运行 'pip install ultralytics'")
        return None
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        return None


def extract_patches_from_nii(image_path: str, mask_path: Optional[str] = None, 
                           patch_size: int = 128, stride_hw: int = 128, 
                           depth_channel: int = 3, logger=None) -> Tuple[List[np.ndarray], List[np.ndarray], dict]:
    """
    从.nii.gz文件中提取patches，不使用Z轴间隔（与原代码不同）
    
    Args:
        image_path: CT图像路径
        mask_path: 标签路径（可选）
        patch_size: patch大小
        stride_hw: 滑动窗口步长
        depth_channel: 2.5D深度通道数
    
    Returns:
        patches: 图像patches列表
        mask_patches: 标签patches列表（如果有）
        metadata: 包含patch位置信息的元数据
    """
    header = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(header)
    mask = None
    if mask_path and os.path.isfile(mask_path):
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    
    spacing = header.GetSpacing()
    z_spacing = spacing[2]
    
    depth, height, width = image.shape
    
    # 不使用Z轴间隔，而是处理所有切片
    extract_ch = list(range(depth))
    
    patches = []
    mask_patches = []
    metadata = {
        'patch_info': [],  # 存储每个patch的信息
        'original_shape': (depth, height, width),
        'spacing': spacing
    }
    
    # 遍历所有切片
    for ch in extract_ch:
        # 创建2.5D图像（当前通道及其相邻通道）
        tiff_image = np.zeros((height, width, depth_channel), dtype=np.float32)
        
        if ch == 0:
            re_ch = [0, 1, 2] if depth > 2 else [0] * depth_channel
            for z, idx in enumerate(re_ch[:depth_channel]):
                tiff_image[:, :, z] = image[idx, :, :]
        elif ch == depth - 1:
            start_ch = max(0, depth - depth_channel)
            re_ch = list(range(start_ch, depth))
            for z, idx in enumerate(range(max(0, ch-depth_channel+1), ch+1)):
                idx = np.clip(idx, 0, depth-1)
                tiff_image[:, :, z % depth_channel] = image[idx, :, :]
        else:
            for z, idx in enumerate(range(ch-1, ch+2)):
                idx = np.clip(idx, 0, depth-1)
                tiff_image[:, :, z] = image[idx, :, :]

        # HU窗口化（Hounsfield Units）
        np.clip(tiff_image, -1000, 400, out=tiff_image)
        # 归一化到0-255
        tiff_image = ((tiff_image + 1000) / 1400 * 255).astype('uint8')

        # 使用滑动窗口提取patch
        for y in range(0, height - patch_size + 1, stride_hw):
            for x in range(0, width - patch_size + 1, stride_hw):
                # 生成patch名称
                patch_name = f"{os.path.basename(image_path).replace('.nii.gz','')}_slice{ch}_y{y}_x{x}"
                
                # 提取图像patch
                patch_img = tiff_image[y:y+patch_size, x:x+patch_size, :]
                patches.append(patch_img)
                
                # 提取标签patch（如果有）
                if mask is not None:
                    try:
                        patch_mask = mask[ch, y:y+patch_size, x:x+patch_size]
                    except Exception as e:
                        logger.error(f"提取标签patch时出错: {e}")
                        patch_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                    mask_patches.append(patch_mask)
                    
                    # 记录patch信息
                    metadata['patch_info'].append({
                        'patch_name': patch_name,
                        'slice_idx': ch,
                        'y_start': y,
                        'x_start': x,
                        'has_positive': np.sum(patch_mask) > 0
                    })
                else:
                    # 记录patch信息
                    metadata['patch_info'].append({
                        'patch_name': patch_name,
                        'slice_idx': ch,
                        'y_start': y,
                        'x_start': x,
                        'has_positive': False
                    })
    
    return patches, mask_patches, metadata


def predict_on_patches(patches: List[np.ndarray], model, device='cpu', batch_size=8, logger=None)  -> List:
    """
    使用YOLO模型对patches进行预测
    
    Args:
        patches: 图像patches列表
        model: YOLO模型
        device: 计算设备
        batch_size: 批次大小
    
    Returns:
        predictions: 预测结果列表（YOLO格式）
    """
    if model is None:
        # 如果没有提供模型，返回空预测结果
        logger.warning("警告：没有提供模型，返回空预测结果")
        return [None] * len(patches)
    
    predictions = []
    
    # 将patches按batch_size分批处理
    for i in tqdm(range(0, len(patches), batch_size), desc="预测patches进度"):
        batch_patches = patches[i:i + batch_size]
        
        # 准备批量输入
        batch_inputs = []
        for patch in batch_patches:
            # 确保patch是RGB格式 (YOLO期望输入为彩色图像)
            if patch.shape[2] == 1:  # 单通道灰度图
                patch_rgb = cv2.cvtColor(patch[:, :, 0], cv2.COLOR_GRAY2RGB)
            elif patch.shape[2] == 3:  # 三通道
                patch_rgb = patch
            else:  # 更多通道的情况，只取前三通道
                patch_rgb = patch[:, :, :3]
            batch_inputs.append(patch_rgb)
        
        # 使用YOLO模型进行批量预测
        try:
            results = model(batch_inputs, iou=0.5, conf=0.1, verbose=False)  # 设置verbose=False减少输出
            # 确保results是一个列表
            if not isinstance(results, list):
                results = [results]
            predictions.extend(results)
        except Exception as e:
            logger.error(f"预测过程中出现错误: {e}")
            # 如果批量预测失败，则逐个预测
            for patch in batch_inputs:
                try:
                    result = model(patch, iou=0.5, conf=0.1, verbose=False)
                    predictions.append(result)
                except Exception as e2:
                    logger.error(f"单个预测也失败: {e2}")
                    predictions.append(None)
    
    return predictions


def convert_predictions_to_detection_format(predictions: List, 
                                          metadata: dict) -> List[dict]:
    """
    将YOLO预测结果转换为检测格式，用于FROC计算
    
    Args:
        predictions: YOLO预测结果列表
        metadata: patch元数据
    
    Returns:
        detections: 检测结果列表
    """
    detections = []
    
    for pred, patch_info in zip(predictions, metadata['patch_info']):
        if pred is None or len(pred) == 0:
            continue
        
        # 获取YOLO预测结果
        result = pred[0]  # 获取第一个结果
        
        if result.boxes is not None:
            # 遍历所有检测框
            for box in result.boxes:
                # 获取边界框坐标 (xyxy格式)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()  # 置信度
                cls = int(box.cls.item())  # 类别
                
                # 计算中心点和宽高
                center_x = (x1 + x2) / 2.0 / 128.0  # 归一化到[0,1]
                center_y = (y1 + y2) / 2.0 / 128.0  # 归一化到[0,1]
                width = (x2 - x1) / 128.0
                height = (y2 - y1) / 128.0
                
                detection = {
                    'image_id': patch_info['patch_name'],
                    'class': cls,
                    'x_center': float(center_x),
                    'y_center': float(center_y),
                    'width': float(width),
                    'height': float(height),
                    'confidence': float(conf),
                    'slice_idx': patch_info['slice_idx'],
                    'y_start': patch_info['y_start'],
                    'x_start': patch_info['x_start']
                }
                detections.append(detection)
    
    return detections


def load_ground_truth_from_masks(mask_patches: List[np.ndarray], 
                               metadata: dict) -> List[dict]:
    """
    从mask patches中提取真实标签
    
    Args:
        mask_patches: 标签patches列表
        metadata: patch元数据
    
    Returns:
        ground_truths: 真实标签列表
    """
    ground_truths = []
    
    for mask_patch, patch_info in zip(mask_patches, metadata['patch_info']):
        if np.sum(mask_patch) > 0:  # 只处理包含正样本的patch
            # 使用连通组件分析找到真实标注区域
            _, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_patch > 0).astype(np.uint8))
            
            # 跳过背景组件 (标签0)
            for j in range(1, labels.max() + 1):
                area = stats[j, cv2.CC_STAT_AREA]
                if area > 10:  # 过滤小区域
                    center_x = centroids[j][0] / mask_patch.shape[1]  # 归一化
                    center_y = centroids[j][1] / mask_patch.shape[0]  # 归一化
                    width = stats[j, cv2.CC_STAT_WIDTH] / mask_patch.shape[1]
                    height = stats[j, cv2.CC_STAT_HEIGHT] / mask_patch.shape[0]
                    
                    gt = {
                        'image_id': patch_info['patch_name'],
                        'class': 0,  # 类别
                        'x_center': center_x,
                        'y_center': center_y,
                        'width': width,
                        'height': height,
                        'slice_idx': patch_info['slice_idx'],
                        'y_start': patch_info['y_start'],
                        'x_start': patch_info['x_start']
                    }
                    ground_truths.append(gt)
    
    return ground_truths


def calculate_iou_2d(box1: dict, box2: dict) -> float:
    """
    计算两个2D边界框的IoU
    
    Args:
        box1, box2: 边界框字典，包含 x_center, y_center, width, height
    
    Returns:
        IoU值
    """
    # 将归一化的坐标转换为绝对坐标（假设图像尺寸为patch_size）
    patch_size = 128  # 根据实际情况调整
    
    x1_min = int((box1['x_center'] - box1['width'] / 2) * patch_size)
    y1_min = int((box1['y_center'] - box1['height'] / 2) * patch_size)
    x1_max = int((box1['x_center'] + box1['width'] / 2) * patch_size)
    y1_max = int((box1['y_center'] + box1['height'] / 2) * patch_size)
    
    x2_min = int((box2['x_center'] - box2['width'] / 2) * patch_size)
    y2_min = int((box2['y_center'] - box2['height'] / 2) * patch_size)
    x2_max = int((box2['x_center'] + box2['width'] / 2) * patch_size)
    y2_max = int((box2['y_center'] + box2['height'] / 2) * patch_size)
    
    # 计算交集面积
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # 计算并集面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    # 避免除零错误
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def match_detections_3d(predictions: List[dict], ground_truths: List[dict], 
                       iou_threshold: float = 0.5) -> Tuple[List, List, List]:
    """
    3D匹配预测结果与真实标签（考虑切片连续性）
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标签列表
        iou_threshold: IoU阈值
    
    Returns:
        matched_pairs: 匹配对列表
        unmatched_preds: 未匹配的预测
        unmatched_gts: 未匹配的真实标签
    """
    matched_pairs = []
    unmatched_preds = [p for p in predictions]
    unmatched_gts = [gt for gt in ground_truths]
    
    # 按置信度降序排列预测框
    sorted_pred_indices = sorted(range(len(predictions)), 
                                key=lambda i: predictions[i]['confidence'], 
                                reverse=True)
    
    for pred_idx in tqdm(sorted_pred_indices, desc="匹配预测与真实标签进度"):
        pred = predictions[pred_idx]
        
        best_gt_idx = -1
        best_iou = 0
        
        # 寻找最佳匹配的真实标签（在同一或相邻切片上）
        for gt_idx, gt in enumerate(ground_truths):
            # 检查是否在相邻切片上
            slice_distance = abs(pred['slice_idx'] - gt['slice_idx'])
            if slice_distance <= 1:  # 允许相邻切片匹配
                iou = calculate_iou_2d(pred, gt)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_gt_idx != -1 and best_gt_idx < len(unmatched_gts):
            # 找到匹配
            matched_pairs.append({
                'pred': pred,
                'gt': ground_truths[best_gt_idx],
                'confidence': pred['confidence'],
                'iou': best_iou
            })
            
            # 从未匹配列表中移除
            if pred in unmatched_preds:
                unmatched_preds.remove(pred)
            if ground_truths[best_gt_idx] in unmatched_gts:
                unmatched_gts.remove(ground_truths[best_gt_idx])
    
    return matched_pairs, unmatched_preds, unmatched_gts


def calculate_froc_curve_3d(matched_pairs: List, unmatched_preds: List, 
                          total_cases: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    计算3D FROC曲线
    
    Args:
        matched_pairs: 匹配的预测-真实对
        unmatched_preds: 未匹配的预测（FP）
        total_cases: 总病例数
    
    Returns:
        thresholds: 阈值数组
        sensitivities: 敏感度数组
        fp_per_case: 每个病例的假阳性数
        sensitivity_at_100fps: 平均每个病人100FP时的敏感度
        threshold_at_100fps: 平均每个病人100FP时的阈值
    """
    # 收集所有置信度分数
    confidences = [pair['confidence'] for pair in matched_pairs]
    confidences.extend([pred['confidence'] for pred in unmatched_preds])
    
    if len(confidences) == 0:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0
    
    # 按置信度排序
    sorted_indices = np.argsort(confidences)[::-1]  # 降序排列
    sorted_confidences = np.array(confidences)[sorted_indices]
    
    # 去除重复的置信度值
    unique_confidences = np.unique(sorted_confidences)
    
    # 计算每个阈值下的TP、FP
    sensitivities = []
    fp_per_case_values = []
    thresholds = []
    
    # 统计总的真阳性和假阳性数量
    total_tps = len(matched_pairs)
    total_fps = len(unmatched_preds)
    total_cases = max(total_cases, 1)  # 避免除零错误
    
    for threshold in tqdm(unique_confidences, desc="计算FROC曲线进度"):
        # 计算高于阈值的TP和FP
        tp_at_thresh = sum(1 for pair in matched_pairs if pair['confidence'] >= threshold)
        fp_at_thresh = sum(1 for pred in unmatched_preds if pred['confidence'] >= threshold)
        
        sensitivity = tp_at_thresh / max(total_tps, 1)  # 避免除零错误
        fp_per_case = fp_at_thresh / total_cases
        
        sensitivities.append(sensitivity)
        fp_per_case_values.append(fp_per_case)
        thresholds.append(threshold)
    
    # 计算平均每个病人100FP时的敏感度和对应阈值
    sensitivity_at_100fps = 0.0
    threshold_at_100fps = 0.0
    
    # 将列表转为numpy数组以便处理
    thresholds_array = np.array(thresholds)
    sensitivities_array = np.array(sensitivities)
    fp_per_case_array = np.array(fp_per_case_values)
    
    # 查找最接近100FP/case的点
    if len(fp_per_case_array) > 0:
        # 找到大于等于100FP/case的最小值点
        indices_around_100 = np.where(fp_per_case_array <= 100)[0]
        if len(indices_around_100) > 0:
            # 选择最接近100FP/case的点
            closest_idx = indices_around_100[np.argmin(np.abs(fp_per_case_array[indices_around_100] - 100))]
            sensitivity_at_100fps = sensitivities_array[closest_idx]
            threshold_at_100fps = thresholds_array[closest_idx]
        else:
            # 如果没有小于等于100的点，选择最大的FP/case对应的点
            max_fp_idx = np.argmax(fp_per_case_array)
            sensitivity_at_100fps = sensitivities_array[max_fp_idx]
            threshold_at_100fps = thresholds_array[max_fp_idx]
    
    return thresholds_array, sensitivities_array, fp_per_case_array, sensitivity_at_100fps, threshold_at_100fps


def process_fold_data(csv_path: str, fold_number: int, model_path: Optional[str] = None, 
                     output_dir: str = "./froc_results", iou_threshold: float = 0.5, batch_size: int = 8):
    """
    处理指定折的数据，进行预测并计算FROC指标
    
    Args:
        csv_path: CSV文件路径，包含fold和path列
        fold_number: 要处理的折数
        model_path: 模型文件路径（可选）
        output_dir: 输出目录
        iou_threshold: IoU匹配阈值
        batch_size: 批次大小
    """
    logger = get_logger(output_dir)
    logger.info(f"开始处理第 {fold_number} 折数据...")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 筛选指定折的数据
    fold_data = df[df['fold'] == fold_number]
    logger.info(f"第 {fold_number} 折包含 {len(fold_data)} 个样本")
    
    if len(fold_data) == 0:
        logger.warning(f"警告：第 {fold_number} 折没有数据")
        return
    
    # 根据fold_number构建默认模型路径
    if model_path is None:
        model_path = Path(f"Train_result/Mucus_249_neg_0/Train_20260302_fold{fold_number}/weights/best.pt")
        logger.info(f"使用默认模型路径: {model_path}")
    
    # 加载模型
    model = load_model(str(model_path), logger) if model_path else None
    
    all_predictions = []
    all_ground_truths = []
    total_patients = len(fold_data)
    # 为每个CT图像提取patches并预测
    for idx, row in fold_data.iterrows():
        
        image_path = row['path']
        logger.info(f"{idx + 1}/{total_patients} 处理图像: {os.path.basename(image_path)}")
        
        # 构建mask路径（将image替换为mask）
        mask_path = image_path.replace('/image/', '/mask/')
        
        # 提取patches
        patches, mask_patches, metadata = extract_patches_from_nii(
            image_path, mask_path, logger=logger
        )
        
        logger.info(f"  提取了 {len(patches)} 个patches")
        
        # 模型预测
        predictions = predict_on_patches(patches, model, batch_size=batch_size, logger=logger)
        
        # 转换预测结果为检测格式
        detections = convert_predictions_to_detection_format(predictions, metadata)
        all_predictions.extend(detections)
        
        # 加载真实标签
        if mask_patches:
            ground_truths = load_ground_truth_from_masks(mask_patches, metadata)
            all_ground_truths.extend(ground_truths)
    
    # 匹配预测结果和真实标签
    matched_pairs, unmatched_preds, unmatched_gts = match_detections_3d(
        all_predictions, all_ground_truths, iou_threshold=iou_threshold
    )
    
    logger.info(f"匹配结果: TP={len(matched_pairs)}, FP={len(unmatched_preds)}, FN={len(unmatched_gts)}")
    
    # 计算FROC曲线
    thresholds, sensitivities, fp_per_case, sensitivity_at_100fps, threshold_at_100fps = calculate_froc_curve_3d(
        matched_pairs, unmatched_preds, total_patients
    )
    
    if len(thresholds) == 0:
        logger.warning(f"第 {fold_number} 折: 没有足够的数据计算FROC曲线")
        return
    
    # 计算AUC
    froc_auc = auc(fp_per_case, sensitivities)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存FROC曲线图
    plt.figure(figsize=(10, 8))
    plt.plot(fp_per_case, sensitivities, linewidth=2, label=f'Fold {fold_number}')
    plt.xlabel('FPs per case')
    plt.ylabel('Sensitivity')
    plt.title(f'3D FROC Curve - Fold {fold_number}\nAUC = {froc_auc:.3f}\nSensitivity @ 100 FPS/case = {sensitivity_at_100fps:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    froc_plot_path = os.path.join(output_dir, f'froc_fold_{fold_number}.png')
    plt.savefig(froc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细结果
    results = {
        'fold': fold_number,
        'froc_auc': froc_auc,
        'sensitivity_at_100fps': sensitivity_at_100fps,
        'threshold_at_100fps': threshold_at_100fps,
        'total_patients': total_patients,
        'total_predictions': len(all_predictions),
        'total_ground_truths': len(all_ground_truths),
        'total_matches': len(matched_pairs),
        'total_fps': len(unmatched_preds),
        'total_fns': len(unmatched_gts),
        'detailed_results': {
            'matched_pairs': [dict(pair) for pair in matched_pairs],
            'unmatched_predictions': [dict(pred) for pred in unmatched_preds],
            'unmatched_ground_truths': [dict(gt) for gt in unmatched_gts]
        }
    }
    
    # 保存结果到JSON文件
    results_path = os.path.join(output_dir, f'results_fold_{fold_number}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"FROC分析完成!")
    logger.info(f"AUC: {froc_auc:.3f}")
    logger.info(f"平均每个病人100FP时的敏感度: {sensitivity_at_100fps:.3f}")
    logger.info(f"平均每个病人100FP时的阈值: {threshold_at_100fps:.3f}")
    logger.info(f"结果保存到: {results_path}")
    logger.info(f"FROC曲线图保存到: {froc_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="处理指定折的CT数据，进行预测并计算FROC指标")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV文件路径，包含fold和path列")
    parser.add_argument("--fold_number", type=int, required=True,
                        help="要处理的折数")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型文件路径（可选），如果不提供将使用默认路径")
    parser.add_argument("--output_dir", type=str, default="./froc_results",
                        help="输出目录")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU匹配阈值")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="预测时的批次大小")
    
    args = parser.parse_args()
    
    process_fold_data(
        csv_path=args.csv_path,
        fold_number=args.fold_number,
        model_path=args.model_path,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()