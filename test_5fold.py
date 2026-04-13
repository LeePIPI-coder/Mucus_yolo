import cv2
import os
import numpy as np
import argparse
import tqdm
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from utils.utils import load_dicom_series, pre_processing, post_processing, coord_vox2pat
import ast
from test import predict


def find_depth_dir(dicom_path):
    if "阳性数据" in dicom_path or "广医粘液栓标注" in dicom_path:
        for item in os.listdir(dicom_path):
            if not os.path.isdir(os.path.join(dicom_path, item)):
                continue
            for sub_item in os.listdir(os.path.join(dicom_path, item)):
                if os.path.isdir(os.path.join(dicom_path, item, sub_item)):
                    for sub_sub_item in os.listdir(os.path.join(dicom_path, item, sub_item)):
                        if os.path.isdir(os.path.join(dicom_path, item, sub_item, sub_sub_item)):
                            return os.path.join(dicom_path, item, sub_item, sub_sub_item)
    else:
        for item in os.listdir(dicom_path):
            if not os.path.isdir(os.path.join(dicom_path, item)):
                continue
            for sub_item in os.listdir(os.path.join(dicom_path, item)):
                if os.path.isdir(os.path.join(dicom_path, item, sub_item)):
                    return os.path.join(dicom_path, item, sub_item)
def process_fold_predictions(csv_path, base_model_path_pattern, output_dir="./predictions_by_fold"):
    """
    根据CSV中的fold信息加载对应的模型权重进行预测，并将结果按折数保存到不同的表格中
    
    Args:
        csv_path: 包含fold和dicom_path信息的CSV文件路径
        base_model_path_pattern: 模型路径模式，如 "Train_result/Mucus_249_neg_0/Train_20260302_fold{fold}/weights/best.pt"
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    
    # 获取所有的折数
    folds = df['val_fold'].unique()
    
    print(f"发现 {len(folds)} 个不同的折: {folds}")
    
    # 对每个折进行处理
    for fold in folds:
        print(f"\n开始处理第 {fold} 折...")
        
        # 获取当前折的所有数据
        fold_data = df[df['val_fold'] == fold]
        print(f"第 {fold} 折包含 {len(fold_data)} 个数据样本")
        
        # 构建当前折的模型路径
        model_path = base_model_path_pattern.format(fold=fold)
        
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在: {model_path}")
            print(f"跳过第 {fold} 折的处理")
            continue
        
        print(f"使用模型: {model_path}")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 存储当前折所有预测结果的列表
        all_fold_results = []
        
        # 对当前折的每个数据进行预测
        
        for idx, row in fold_data.iterrows():

            dicom_path = row['dicom_path']
            
            if not os.path.exists(dicom_path):
                print(f"警告: DICOM路径不存在: {dicom_path}")
                continue
            
            print(f"正在预测: {dicom_path}")
            
            try:
                # 执行预测
                        
                dicom_dir = find_depth_dir(dicom_path)
                result_df = predict(dicom_dir, model)
                
                # 添加患者信息和折数标识
                result_df['patient_key'] = row['patient_key']
                result_df['val_fold'] = fold
                
                all_fold_results.append(result_df)
                
            except Exception as e:
                print(f"预测 {dicom_path} 时出错: {str(e)}")
                continue
        
        # 合并当前折的所有预测结果
        if all_fold_results:
            combined_results = pd.concat(all_fold_results, ignore_index=True)
            
            # 保存当前折的结果到单独的CSV文件
            output_file = os.path.join(output_dir, f"predictions_fold_{fold}.csv")
            combined_results.to_csv(output_file, index=False)
            print(f"第 {fold} 折的预测结果已保存到: {output_file}")
            print(f"共 {len(combined_results)} 条预测结果")
        else:
            print(f"第 {fold} 折没有成功生成任何预测结果")
    
    print("\n所有折的预测完成！")


def main():
    parser = argparse.ArgumentParser(description="根据折数加载对应模型权重进行预测")
    parser.add_argument("--csv_path", type=str, 
                        default="/data/yolo_dataset_249/Kfold_neg_0/5_scan_data.csv",
                        help="CSV文件路径，包含val_fold和dicom_path列")
    parser.add_argument("--base_model_path", type=str,
                        default="Train_result/Mucus_249_neg_0/Train_fold{fold}/weights/best.pt",
                        help="基础模型路径模式，其中{fold}会被替换为具体的折数")
    parser.add_argument("--output_dir", type=str, 
                        default="./predictions_by_fold/5-scan",
                        help="输出目录")
    
    args = parser.parse_args()
    
    print(f"CSV路径: {args.csv_path}")
    print(f"基础模型路径: {args.base_model_path}")
    print(f"输出目录: {args.output_dir}")
    
    # 确认CSV文件存在
    if not os.path.exists(args.csv_path):
        print(f"错误: CSV文件不存在: {args.csv_path}")
        return
    
    # 执行预测
    process_fold_predictions(args.csv_path, args.base_model_path, args.output_dir)


if __name__ == "__main__":
    main()