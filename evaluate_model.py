import os
import numpy as np
import cv2
import cc3d
from tqdm import tqdm
import argparse
import json


def load_2d_labels(label_dir, image_dir):
    """
    加载2D标签框
    
    Args:
        label_dir (str): 标签文件目录
        image_dir (str): 图像文件目录
    
    Returns:
        dict: 键为z坐标，值为该z层的2D标签框列表
    """
    # 获取所有图像文件并排序
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".npy")])
    
    # 存储2D标签框
    labels_2d = {}
    
    for img_file in tqdm(image_files, desc="Loading 2D labels"):
        # 构建标签文件路径
        txt_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        
        if not os.path.exists(txt_path):
            continue
        
        # 解析z坐标（假设文件名包含z信息）
        try:
            # 尝试从文件名中提取z值
            z = int(img_file.split("z")[1].split("_")[0])
        except:
            # 如果文件名没有z信息，使用索引作为z值
            z = len(labels_2d)
        
        # 读取标签文件
        boxes = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # YOLO格式: class_id x_center y_center width height
                    class_id, x_center, y_center, width, height = map(float, parts)
                    
                    # 转换为像素坐标 (x1, y1, x2, y2)
                    # 假设图像尺寸为512x512
                    img_width, img_height = 512, 512
                    x1 = (x_center - width/2) * img_width
                    y1 = (y_center - height/2) * img_height
                    x2 = (x_center + width/2) * img_width
                    y2 = (y_center + height/2) * img_height
                    
                    boxes.append([x1, y1, z, x2, y2, z])
        
        if boxes:
            labels_2d[z] = boxes
    
    return labels_2d


def merge_2d_labels_to_3d(labels_2d, iou_threshold=0.5):
    """
    将2D标签框合并为3D标签框
    
    Args:
        labels_2d (dict): 键为z坐标，值为该z层的2D标签框列表
        iou_threshold (float): IoU阈值，用于判断是否合并
    
    Returns:
        list: 3D标签框列表
    """
    if not labels_2d:
        return []
    
    # 收集所有2D标签框
    all_boxes = []
    for z, boxes in labels_2d.items():
        all_boxes.extend(boxes)
    
    # 按z值排序
    sorted_boxes = sorted(all_boxes, key=lambda x: x[2])
    
    # 合并为3D标签框
    merged_boxes = []
    
    for box in sorted_boxes:
        merged = False
        # 检查所有已存在的3D框，看是否以z-1结束且IoU>阈值
        for i, merged_box in enumerate(merged_boxes):
            if merged_box[5] == box[2] - 1:  # 3D框结束于z-1
                iou = compute_iou(box, merged_box)
                if iou > iou_threshold:
                    # 合并：更新3D框的坐标以包含当前框
                    new_x1 = min(merged_box[0], box[0])
                    new_y1 = min(merged_box[1], box[1])
                    new_x2 = max(merged_box[3], box[3])
                    new_y2 = max(merged_box[4], box[4])
                    new_z2 = box[2]
                    merged_boxes[i] = [new_x1, new_y1, merged_box[2], new_x2, new_y2, new_z2]
                    merged = True
                    break
        if not merged:
            # 创建新的3D框
            merged_boxes.append([box[0], box[1], box[2], box[3], box[4], box[2]])
    
    return merged_boxes


def load_3d_mask_and_compute_bbox(mask_dir, image_dir):
    """
    加载3D掩码并计算外接3D矩形框
    
    Args:
        mask_dir (str): 掩码文件目录
        image_dir (str): 图像文件目录
    
    Returns:
        list: 3D标签框列表
    """
    # 获取所有图像文件并排序
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".npy")])
    
    # 加载所有掩码并构建3D掩码
    masks = []
    z_indices = []
    
    for img_file in tqdm(image_files, desc="Loading masks"):
        # 构建掩码文件路径
        mask_path = os.path.join(mask_dir, os.path.splitext(img_file)[0] + ".png")
        
        if not os.path.exists(mask_path):
            continue
        
        # 加载掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
        
        # 解析z坐标
        try:
            z = int(img_file.split("z")[1].split("_")[0])
        except:
            z = len(z_indices)
        z_indices.append(z)
    
    if not masks:
        return []
    
    # 构建3D掩码
    masks_3d = np.stack(masks, axis=0)
    
    # 连通组件分析
    binary_mask = (masks_3d > 0).astype(np.uint8)
    labels_out = cc3d.connected_components(binary_mask, connectivity=8)
    
    # 为每个连通组件计算外接3D矩形框
    boxes_3d = []
    num_objects = labels_out.max()
    
    for obj_label in range(1, num_objects + 1):
        # 提取当前目标的掩码区域
        obj_mask = (labels_out == obj_label).astype(np.uint8)
        
        # 找到所有非零像素的坐标
        coords = np.where(obj_mask > 0)
        z_coords, y_coords, x_coords = coords
        
        # 计算外接矩形框
        x1 = np.min(x_coords)
        y1 = np.min(y_coords)
        z1 = np.min(z_coords)
        x2 = np.max(x_coords)
        y2 = np.max(y_coords)
        z2 = np.max(z_coords)
        
        # 转换为与预测框相同的格式 [x1, y1, z1, x2, y2, z2]
        boxes_3d.append([x1, y1, z1, x2, y2, z2])
    
    return boxes_3d


def compute_iou(box1, box2):
    """计算两个2D检测框的IoU（忽略z轴）"""
    # box1: [x1, y1, z, x2, y2, z]
    # box2: [x1, y1, z, x2, y2, z]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[3] - box1[0]) * (box1[4] - box1[1])
    area2 = (box2[3] - box2[0]) * (box2[4] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def compute_3d_iou(box1, box2):
    """计算两个3D检测框的IoU"""
    # box1: [x1, y1, z1, x2, y2, z2]
    # box2: [x1, y1, z1, x2, y2, z2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    z1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    z2 = min(box1[5], box2[5])
    if x2 <= x1 or y2 <= y1 or z2 <= z1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1) * (z2 - z1)
    volume1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    volume2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union = volume1 + volume2 - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    评估模型预测性能
    
    Args:
        pred_boxes (list): 模型预测的3D检测框列表
        gt_boxes (list): 真实的3D标签框列表
        iou_threshold (float): IoU阈值，用于判断是否检测正确
    
    Returns:
        dict: 性能指标
    """
    # 初始化
    tp = 0  # 真正例
    fp = 0  # 假正例
    fn = len(gt_boxes)  # 假负例
    
    # 标记已匹配的真实框
    matched_gt = [False] * len(gt_boxes)
    
    # 对每个预测框，寻找最佳匹配的真实框
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(gt_boxes):
            if not matched_gt[i]:
                iou = compute_3d_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        if best_iou >= iou_threshold:
            # 检测正确
            tp += 1
            fn -= 1
            matched_gt[best_gt_idx] = True
        else:
            # 检测错误
            fp += 1
    
    # 计算性能指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def load_predicted_boxes_from_json(json_path):
    """
    从Upload_dicom.py生成的JSON文件中加载3D检测框
    
    Args:
        json_path (str): JSON文件路径
    
    Returns:
        list: 3D预测框列表
    """
    pred_boxes = []
    
    if not os.path.exists(json_path):
        print(f"Warning: JSON file not found: {json_path}")
        return pred_boxes
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
        # 解析JSON结构
        try:
            lesions = data.get('study', [])[0].get('series', [])[0].get('lesion', [])
            for lesion in lesions:
                # 提取检测框信息
                # 注意：这里需要根据实际的JSON结构进行调整
                # 示例：从lesion字典中提取坐标信息
                # 假设检测框信息存储在特定字段中
                # 这里需要根据实际实现进行修改
                pass
        except Exception as e:
            print(f"Error parsing JSON: {e}")
    
    return pred_boxes


def load_predicted_boxes(pred_dir):
    """
    加载模型预测的3D检测框
    
    Args:
        pred_dir (str): 预测结果目录
    
    Returns:
        list: 3D预测框列表
    """
    pred_boxes = []
    
    # 遍历预测结果文件
    for file in os.listdir(pred_dir):
        if file.endswith(".json"):
            json_path = os.path.join(pred_dir, file)
            pred_boxes.extend(load_predicted_boxes_from_json(json_path))
    
    return pred_boxes


def simulate_predicted_boxes():
    """
    模拟生成3D预测框（用于测试）
    
    Returns:
        list: 模拟的3D预测框列表
    """
    # 模拟一些3D检测框
    # 格式：[x1, y1, z1, x2, y2, z2]
    return [
        [100, 100, 5, 150, 150, 10],
        [200, 200, 15, 250, 250, 20],
        [300, 300, 25, 350, 350, 30]
    ]


def main():
    parser = argparse.ArgumentParser(description="评估模型性能")
    parser.add_argument("--image_dir", type=str, required=True, help="图像文件目录")
    parser.add_argument("--mask_dir", type=str, required=True, help="掩码文件目录")
    parser.add_argument("--label_dir", type=str, required=True, help="标签文件目录")
    parser.add_argument("--pred_dir", type=str, help="预测结果目录")
    parser.add_argument("--json_path", type=str, help="预测结果JSON文件路径")
    parser.add_argument("--method", type=str, choices=["merge_2d", "3d_mask"], default="merge_2d", 
                        help="生成3D标签框的方法：merge_2d（合并2D标签框）或3d_mask（从3D掩码计算）")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU阈值")
    parser.add_argument("--use_simulated_preds", action="store_true", help="使用模拟的预测框进行测试")
    
    args = parser.parse_args()
    
    # 加载预测框
    print("Loading predicted boxes...")
    if args.use_simulated_preds:
        pred_boxes = simulate_predicted_boxes()
        print(f"Using simulated predictions: {len(pred_boxes)} boxes")
    elif args.json_path:
        pred_boxes = load_predicted_boxes_from_json(args.json_path)
        print(f"Loaded predictions from JSON: {len(pred_boxes)} boxes")
    elif args.pred_dir:
        pred_boxes = load_predicted_boxes(args.pred_dir)
        print(f"Loaded predictions from directory: {len(pred_boxes)} boxes")
    else:
        print("Error: Please specify either --pred_dir, --json_path, or --use_simulated_preds")
        return
    
    # 生成3D标签框
    print("Generating 3D ground truth boxes...")
    if args.method == "merge_2d":
        # 从2D标签框合并
        labels_2d = load_2d_labels(args.label_dir, args.image_dir)
        gt_boxes = merge_2d_labels_to_3d(labels_2d, args.iou_threshold)
        print(f"Generated 3D GT boxes from merged 2D labels: {len(gt_boxes)} boxes")
    else:
        # 从3D掩码计算
        gt_boxes = load_3d_mask_and_compute_bbox(args.mask_dir, args.image_dir)
        print(f"Generated 3D GT boxes from 3D mask: {len(gt_boxes)} boxes")
    
    # 评估性能
    print("Evaluating performance...")
    metrics = evaluate_predictions(pred_boxes, gt_boxes, args.iou_threshold)
    
    # 输出结果
    print("\nEvaluation Results:")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # 保存评估结果
    results = {
        "predicted_boxes": len(pred_boxes),
        "ground_truth_boxes": len(gt_boxes),
        "metrics": metrics
    }
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("\nEvaluation results saved to evaluation_results.json")


if __name__ == "__main__":
    main()