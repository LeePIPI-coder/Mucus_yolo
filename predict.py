import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO

def predict_image(image_path, model):
    """
    对单张图像进行预测
    
    Args:
        image_path: 图像路径
        model: 预测模型
    
    Returns:
        预测结果列表，每个元素为 [class_id, x_center, y_center, width, height, confidence]
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 使用YOLO模型进行预测
    results = model(img, device=0, conf=0.1, verbose=False)

    predictions = []
    
    # 处理预测结果
    for result in results:
        # 遍历每个检测结果
        for box in result.boxes:
            # 提取边界框坐标（YOLO返回的是像素坐标，格式为 [x1, y1, x2, y2]）
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # 计算中心点坐标和宽高
            x_center = (x1 + x2) / 2 / width  # 归一化到0-1范围
            y_center = (y1 + y2) / 2 / height  # 归一化到0-1范围
            w = (x2 - x1) / width  # 归一化到0-1范围
            h = (y2 - y1) / height  # 归一化到0-1范围
            
            # 提取类别ID和置信度
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # 添加到预测结果列表
            predictions.append([class_id, x_center, y_center, w, h, confidence])
    
    return predictions


def process_fold(fold_path, model_path, images_subdir, predictions_subdir, batch_size=16):
    """
    处理单个折的验证集（批量预测）
    
    Args:
        fold_path: 折的路径
        model_path: 模型权重文件路径
        images_subdir: 图像子目录名
        predictions_subdir: 预测结果子目录名
        batch_size: 批量处理大小
    """
    # 构建路径
    images_dir = os.path.join(fold_path, images_subdir)
    predictions_dir = os.path.join(fold_path, predictions_subdir)
    
    # 创建预测结果目录
    os.makedirs(predictions_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(str(model_path))
    
    # 收集所有图像文件路径
    image_paths = []
    base_names = []
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            image_paths.append(img_path)
            base_names.append(base_name)
    
    # 批量处理
    total_images = len(image_paths)
    print(f"开始处理 {total_images} 张图像，批量大小: {batch_size}")
    
    for i in range(0, total_images, batch_size):
        # 获取当前批次的图像路径和基本名称
        batch_paths = image_paths[i:i+batch_size]
        batch_base_names = base_names[i:i+batch_size]
        
        # 读取批次中的图像
        batch_images = []
        valid_indices = []
        for j, img_path in enumerate(batch_paths):
            img = cv2.imread(img_path)
            if img is not None:
                batch_images.append(img)
                valid_indices.append(j)
        
        if not batch_images:
            continue
        
        # 批量预测
        results = model(batch_images, device=0, conf=0.1, verbose=False)
        
        # 处理预测结果
        for j, (result, idx) in enumerate(zip(results, valid_indices)):
            base_name = batch_base_names[idx]
            pred_file = os.path.join(predictions_dir, base_name + '.txt')
            img = batch_images[j]
            height, width = img.shape[:2]
            
            # 处理当前图像的预测结果
            predictions = []
            for box in result.boxes:
                # 提取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 计算中心点坐标和宽高（归一化）
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                
                # 提取类别ID和置信度
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                predictions.append([class_id, x_center, y_center, w, h, confidence])
            
            # 保存预测结果
            with open(pred_file, 'w') as f:
                for pred in predictions:
                    if len(pred) >= 5:
                        class_id = int(pred[0])
                        x_center = float(pred[1])
                        y_center = float(pred[2])
                        width = float(pred[3])
                        height = float(pred[4])
                        confidence = float(pred[5]) if len(pred) > 5 else 1.0
                        f.write(f"{class_id} {x_center} {y_center} {width} {height} {confidence}\n")
    
    print(f"已处理 {fold_path}，预测结果保存到 {predictions_dir}")


def main():
    parser = argparse.ArgumentParser(description="预测程序，生成FROC计算所需的预测结果")
    parser.add_argument("--data_root", type=str, default="/data/yolo_dataset_249/Kfold",
                        help="数据根目录，包含所有fold子目录")
    parser.add_argument("--images_subdir", type=str, default="images/val",
                        help="图像子目录名 (相对于每个fold目录)")
    parser.add_argument("--predictions_subdir", type=str, default="predictions",
                        help="预测结果子目录名 (相对于每个fold目录)")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="折数")
    
    args = parser.parse_args()
    
    # 处理每个折
    for fold_num in range(args.num_folds):
        model_path = Path(f"Train_result/Mucus_249_neg_0/Train_20260302_fold{fold_num}/weights/best.pt")
        fold_path = os.path.join(args.data_root, f"fold{fold_num}")
        
        if not os.path.exists(fold_path):
            print(f"警告: 第 {fold_num} 折目录不存在: {fold_path}")
            continue
        
        images_dir = os.path.join(fold_path, args.images_subdir)
        if not os.path.exists(images_dir):
            print(f"警告: 第 {fold_num} 折缺少图像目录: {images_dir}")
            continue
        
        # 处理当前折
        process_fold(fold_path, model_path, args.images_subdir, args.predictions_subdir)


if __name__ == "__main__":
    main()