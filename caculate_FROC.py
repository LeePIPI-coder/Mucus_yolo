import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
import xml.etree.ElementTree as ET
from sklearn.metrics import auc
import cv2
from collections import defaultdict
import argparse


def load_predictions_and_ground_truth(predictions_dir: str, ground_truth_dir: str, images_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    加载预测结果和真实标签
    
    Args:
        predictions_dir: 预测结果目录（通常是模型输出的检测结果）
        ground_truth_dir: 真实标签目录
        images_dir: 图像目录（用于获取图像尺寸信息）
    
    Returns:
        predictions: 预测结果列表 [base_name, boxes, image_size]
        ground_truths: 真实标签列表 [base_name, boxes, image_size]
    """
    predictions = []
    ground_truths = []
    
    # 获取所有图像文件
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_file)[0]
            pred_file = os.path.join(predictions_dir, base_name + '.txt')
            gt_file = os.path.join(ground_truth_dir, base_name + '.txt')
            
            # 加载预测结果
            pred_boxes = []
            if os.path.exists(pred_file):
                with open(pred_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:  # class, x_center, y_center, width, height
                            class_id, x_center, y_center, width, height = map(float, parts)
                            confidence = 1.0  # 如果没有置信度分数，设为1.0
                            if len(parts) == 6:  # 如果有置信度分数
                                confidence = float(parts[5])
                            pred_boxes.append({
                                'class': int(class_id),
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height,
                                'confidence': confidence
                            })
            
            # 加载真实标签
            gt_boxes = []
            if os.path.exists(gt_file):
                with open(gt_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class, x_center, y_center, width, height
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            gt_boxes.append({
                                'class': int(class_id),
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height
                            })
            
            # 获取图像尺寸
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
            else:
                # 默认尺寸，如果无法加载图像
                height, width = 512, 512
            
            predictions.append({
                'image_id': base_name,
                'boxes': pred_boxes,
                'image_size': (width, height)
            })
            
            ground_truths.append({
                'image_id': base_name,
                'boxes': gt_boxes,
                'image_size': (width, height)
            })
    
    return predictions, ground_truths


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        box1, box2: 边界框字典，包含 x_center, y_center, width, height
    
    Returns:
        IoU值
    """
    # 将归一化的坐标转换为绝对坐标
    x1_min = (box1['x_center'] - box1['width'] / 2) * box1.get('image_width', 1)
    y1_min = (box1['y_center'] - box1['height'] / 2) * box1.get('image_height', 1)
    x1_max = (box1['x_center'] + box1['width'] / 2) * box1.get('image_width', 1)
    y1_max = (box1['y_center'] + box1['height'] / 2) * box1.get('image_height', 1)
    
    x2_min = (box2['x_center'] - box2['width'] / 2) * box2.get('image_width', 1)
    y2_min = (box2['y_center'] - box2['height'] / 2) * box2.get('image_height', 1)
    x2_max = (box2['x_center'] + box2['width'] / 2) * box2.get('image_width', 1)
    y2_max = (box2['y_center'] + box2['height'] / 2) * box2.get('image_height', 1)
    
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


def match_detections(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> Tuple[List, List, List]:
    """
    匹配预测结果与真实标签
    
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
    unmatched_preds = []
    unmatched_gts = []
    
    # 为每个图像匹配预测和真实标签
    for pred_img, gt_img in zip(predictions, ground_truths):
        assert pred_img['image_id'] == gt_img['image_id'], "图像ID不匹配"
        
        pred_boxes = pred_img['boxes']
        gt_boxes = gt_img['boxes']
        
        # 添加图像尺寸信息到边界框
        for box in pred_boxes:
            box['image_width'] = pred_img['image_size'][0]
            box['image_height'] = pred_img['image_size'][1]
        
        for box in gt_boxes:
            box['image_width'] = gt_img['image_size'][0]
            box['image_height'] = gt_img['image_size'][1]
        
        # 初始化匹配状态
        pred_matched = [False] * len(pred_boxes)
        gt_matched = [False] * len(gt_boxes)
        
        # 按置信度降序排列预测框
        sorted_pred_indices = sorted(range(len(pred_boxes)), 
                                key=lambda i: pred_boxes[i]['confidence'], 
                                reverse=True)
        
        # 匹配预测框和真实框,进行IOU匹配
        for pred_idx in sorted_pred_indices:
            best_gt_idx = -1
            best_iou = 0
            
            for gt_idx in range(len(gt_boxes)):
                if gt_matched[gt_idx]:
                    continue
                
                iou = calculate_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                # 找到匹配
                matched_pairs.append({
                    'pred': pred_boxes[pred_idx],
                    'gt': gt_boxes[best_gt_idx],
                    'confidence': pred_boxes[pred_idx]['confidence'],
                    'iou': best_iou
                })
                pred_matched[pred_idx] = True
                gt_matched[best_gt_idx] = True
            else:
                # 未匹配的预测（FP）
                unmatched_preds.append({
                    'pred': pred_boxes[pred_idx],
                    'confidence': pred_boxes[pred_idx]['confidence'],
                    'image_id': pred_img['image_id']
                })
        
        # 添加未匹配的真实框（FN）
        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                unmatched_gts.append({
                    'gt': gt_boxes[gt_idx],
                    'image_id': gt_img['image_id']
                })
    
    return matched_pairs, unmatched_preds, unmatched_gts


def calculate_froc_curve(matched_pairs: List, unmatched_preds: List, unmatched_gts: List, 
                        total_cases: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算FROC曲线
    
    Args:
        matched_pairs: 匹配的预测-真实对
        unmatched_preds: 未匹配的预测（FP）
        unmatched_gts: 未匹配的真实标签（FN）
        total_cases: 总病例数
    
    Returns:
        thresholds: 阈值数组
        sensitivities: 敏感度数组
        fp_per_case: 每个病例的假阳性数
    """
    # 收集所有置信度分数
    confidences = [pair['confidence'] for pair in matched_pairs]
    confidences.extend([pred['confidence'] for pred in unmatched_preds])
    
    if len(confidences) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 按置信度排序
    sorted_indices = np.argsort(confidences)[::-1]  # 降序排列
    sorted_confidences = np.array(confidences)[sorted_indices]
    
    # 去除重复的置信度值
    unique_confidences = np.unique(sorted_confidences)
    
    # 计算每个阈值下的TP、FP
    sensitivities = []
    fp_per_case_values = []
    thresholds = []
    
    # 统计总的真阳性和真阴性数量
    total_tps = len(matched_pairs)
    total_fps = len(unmatched_preds)
    total_cases = max(total_cases, 1)  # 避免除零错误
    
    for threshold in unique_confidences:
        # 计算高于阈值的TP和FP
        tp_at_thresh = sum(1 for pair in matched_pairs if pair['confidence'] >= threshold)
        fp_at_thresh = sum(1 for pred in unmatched_preds if pred['confidence'] >= threshold)
        
        sensitivity = tp_at_thresh / max(total_tps, 1)  # 避免除零错误
        fp_per_case = fp_at_thresh / total_cases
        
        sensitivities.append(sensitivity)
        fp_per_case_values.append(fp_per_case)
        thresholds.append(threshold)
    
    return np.array(thresholds), np.array(sensitivities), np.array(fp_per_case_values)


def find_threshold_for_target_fp(fp_per_case: np.ndarray, thresholds: np.ndarray, target_fp: float) -> float:
    """
    找到达到目标FP数的阈值
    
    Args:
        fp_per_case: 每个病例的FP数数组
        thresholds: 阈值数组
        target_fp: 目标FP数
    
    Returns:
        对应的阈值
    """
    if len(fp_per_case) == 0:
        return 0.0
    
    # 找到最接近目标FP数的索引
    diffs = np.abs(fp_per_case - target_fp)
    closest_idx = np.argmin(diffs)
    
    return thresholds[closest_idx]


def extract_patient_info_from_filename(filename: str) -> str:
    """
    从文件名中提取患者信息
    例如: E0001013_20091215_slice140_y32_x32 -> E0001013_20091215
    """
    parts = filename.split('_')
    patient_part = parts[0]
    for part in parts[1:]:
        if 'slice' in part:
            break
        patient_part += '_' + part
    return patient_part


def calculate_froc_for_fold(fold_predictions_dir: str, fold_ground_truth_dir: str, 
                           fold_images_dir: str, fold_num: int, output_dir: str) -> Dict[str, Any]:
    """
    为特定折计算FROC曲线
    
    Args:
        fold_predictions_dir: 该折的预测结果目录
        fold_ground_truth_dir: 该折的真实标签目录
        fold_images_dir: 该折的图像目录
        fold_num: 折数
        output_dir: 输出目录
    
    Returns:
        包含FROC分析结果的字典
    """
    print(f"正在处理第 {fold_num} 折...")
    
    # 加载预测结果和真实标签
    predictions, ground_truths = load_predictions_and_ground_truth(
        fold_predictions_dir, fold_ground_truth_dir, fold_images_dir
    )
    
    # 提取患者信息 得到患者的命名 如:E0001013_20091215
    patients = set()
    for pred in predictions:
        patient_id = extract_patient_info_from_filename(pred['image_id'])
        patients.add(patient_id)
    total_patients = len(patients)
    
    # 匹配检测结果
    matched_pairs, unmatched_preds, unmatched_gts = match_detections(
        predictions, ground_truths, iou_threshold=0.5
    )
    
    print(f"第 {fold_num} 折: {len(predictions)} 张图像, {len(patients)} 个患者")
    print(f"匹配的对数: {len(matched_pairs)}, FP数: {len(unmatched_preds)}, FN数: {len(unmatched_gts)}")
    
    # 计算FROC曲线
    thresholds, sensitivities, fp_per_case = calculate_froc_curve(
        matched_pairs, unmatched_preds, unmatched_gts, total_patients
    )
    
    if len(thresholds) == 0:
        print(f"第 {fold_num} 折: 没有足够的数据计算FROC曲线")
        return {}
    
    # 找到每个患者平均100个FP的阈值
    target_fp_per_case = 100
    threshold_for_100fp = find_threshold_for_target_fp(fp_per_case, thresholds, target_fp_per_case)
    
    # 计算AUC
    froc_auc = auc(fp_per_case, sensitivities)
    
    # 保存FROC曲线图
    plt.figure(figsize=(10, 8))
    plt.plot(fp_per_case, sensitivities, linewidth=2, label=f'Fold {fold_num}')
    plt.xlabel('平均每个病例的假阳性数 (FPs per case)')
    plt.ylabel('敏感度 (Sensitivity)')
    plt.title(f'FROC Curve - Fold {fold_num}\nAUC = {froc_auc:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    froc_plot_path = os.path.join(output_dir, f'froc_fold_{fold_num}.png')
    plt.savefig(froc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 记录FP点信息
    fps_at_threshold = [pred for pred in unmatched_preds if pred['confidence'] >= threshold_for_100fp]
    
    result = {
        'fold': fold_num,
        'threshold_for_100fp': threshold_for_100fp,
        'froc_auc': froc_auc,
        'total_patients': total_patients,
        'total_images': len(predictions),
        'total_matches': len(matched_pairs),
        'total_fps': len(unmatched_preds),
        'total_fns': len(unmatched_gts),
        'fps_at_100fp_threshold': len(fps_at_threshold),
        'fp_details': fps_at_threshold  # 记录FP详细信息
    }
    
    print(f"第 {fold_num} 折: 在阈值 {threshold_for_100fp:.3f} 下，每个患者平均FP数 ≈ {len(fps_at_threshold)/max(total_patients, 1):.2f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate FROC curves for k-fold cross validation")
    parser.add_argument("--data_root", type=str, required=True, 
                        help="数据根目录，包含所有fold子目录")
    parser.add_argument("--predictions_subdir", type=str, default="predictions",
                        help="预测结果子目录名 (相对于每个fold目录)")
    parser.add_argument("--ground_truth_subdir", type=str, default="labels/val",
                        help="真实标签子目录名 (相对于每个fold目录)")
    parser.add_argument("--images_subdir", type=str, default="images/val",
                        help="图像子目录名 (相对于每个fold目录)")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="折数")
    parser.add_argument("--output_dir", type=str, default="./froc_results",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    # 为每个折计算FROC
    for fold_num in range(args.num_folds):
        fold_path = os.path.join(args.data_root, f"fold{fold_num}")
        
        if not os.path.exists(fold_path):
            print(f"警告: 第 {fold_num} 折目录不存在: {fold_path}")
            continue
        
        fold_predictions_dir = os.path.join(fold_path, args.predictions_subdir)
        fold_ground_truth_dir = os.path.join(fold_path, args.ground_truth_subdir)
        fold_images_dir = os.path.join(fold_path, args.images_subdir)
        
        if not all(os.path.exists(d) for d in [fold_predictions_dir, fold_ground_truth_dir, fold_images_dir]):
            print(f"警告: 第 {fold_num} 折缺少必要目录")
            continue
        
        result = calculate_froc_for_fold(
            fold_predictions_dir, fold_ground_truth_dir, 
            fold_images_dir, fold_num, args.output_dir
        )
        
        if result:  # 只有当结果有效时才添加
            all_results.append(result)
    
    # 保存汇总结果
    summary_path = os.path.join(args.output_dir, "froc_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总统计
    if all_results:
        avg_threshold = np.mean([r['threshold_for_100fp'] for r in all_results])
        avg_auc = np.mean([r['froc_auc'] for r in all_results])
        
        print("\n=== FROC分析汇总 ===")
        print(f"平均阈值 (100FP/case): {avg_threshold:.3f}")
        print(f"平均AUC: {avg_auc:.3f}")
        
        for result in all_results:
            print(f"Fold {result['fold']}: 阈值={result['threshold_for_100fp']:.3f}, AUC={result['froc_auc']:.3f}")
    
    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
