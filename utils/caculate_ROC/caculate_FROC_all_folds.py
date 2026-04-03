#!/usr/bin/env python3
"""
批量处理所有折的数据，使用对应的YOLO模型进行预测并计算FROC指标
直接调用FROC.py中的函数
"""

import os
import argparse
from pathlib import Path
from .FROC_2D import process_fold_data  # 直接导入FROC.py中的函数


def run_fold_processing(csv_path: str, fold_number: int, num_folds: int, output_dir: str = "./froc_results-3D"):
    """
    运行单个折的处理
    """
    # 构建模型路径
    model_path = Path(f"Train_result/Mucus_249_neg_0/Train_20260302_fold{fold_number}/weights/best.pt")

    print(f"开始处理第 {fold_number} 折...")
    print(f"模型路径: {model_path}")
    print(f"CSV路径: {csv_path}")
    
    # 检查模型文件是否存在
    if not model_path.exists():
        print(f"警告: 模型文件不存在: {model_path}")
        print(f"将尝试使用默认路径进行处理...")
        process_fold_data(
            csv_path=csv_path,
            fold_number=fold_number,
            model_path=None,  # 使用默认路径
            output_dir=output_dir
        )
    else:
        process_fold_data(
            csv_path=csv_path,
            fold_number=fold_number,
            model_path=str(model_path),
            output_dir=output_dir
        )

def main():
    parser = argparse.ArgumentParser(description="批量处理所有折的数据")
    parser.add_argument("--csv_path", type=str, default="/data/yolo_dataset_249/Kfold/fold_assignment_integrated.csv",
                        help="CSV文件路径，包含fold和path列")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="总折数")
    parser.add_argument("--start_fold", type=int, default=0,
                        help="起始折数")
    parser.add_argument("--output_dir", type=str, default="./froc_results",
                        help="输出目录")
    
    args = parser.parse_args()
    
    print(f"开始批量处理 {args.start_fold} 到 {args.start_fold + args.num_folds - 1} 折的数据...")
    
    for fold_num in range(args.start_fold, args.start_fold + args.num_folds):
        # try:
            run_fold_processing(args.csv_path, fold_num, args.num_folds, args.output_dir)
            print(f"第 {fold_num} 折处理完成")
        # except Exception as e:
        #     print(f"处理第 {fold_num} 折时发生错误: {e}")
        #     continue
    
    print("所有折处理完成!")


if __name__ == "__main__":
    main()