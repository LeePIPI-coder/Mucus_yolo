#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path

def integrate_csv_files(nifti_csv_path, fold_csv_path, output_path=None):
    print("开始整合CSV文件...")
    
    # 1. 读取文件
    nifti_df = pd.read_csv(nifti_csv_path)
    fold_df = pd.read_csv(fold_csv_path)
    
    # 2. 自动识别列名
    path_column = "path" if "path" in nifti_df.columns else nifti_df.columns[0]
    fold_column = fold_df.columns[0]      # 第一列：折数
    filename_column = fold_df.columns[1]  # 第二列：文件名
    
    # 3. 关键步骤：统一匹配键 (Match Key)
    # 做法：取文件名，去掉所有扩展名，并去除前后空格
    def clean_filename(x):
        return os.path.basename(str(x)).split('.')[0].strip()

    nifti_df['match_key'] = nifti_df[path_column].apply(clean_filename)
    fold_df['match_key'] = fold_df[filename_column].apply(clean_filename)
    
    print(f"正在基于关键字段进行匹配...")

    # 4. 执行合并 (Inner Join)
    # 这会自动完成：如果 fold_df 的文件名在 nifti_df 中存在，则保留并合并
    merged_df = pd.merge(
        fold_df, 
        nifti_df[['match_key', path_column]], 
        on='match_key', 
        how='inner'
    )

    # 5. 提取你需要的两列：fold 和 path
    # 我们使用原本识别出的列名，最后再统一重命名
    result_df = merged_df[[fold_column, path_column]].copy()
    result_df.columns = ['fold', 'path']

    # 6. 保存与输出
    if output_path is None:
        output_path = str(Path(fold_csv_path).with_name(f"{Path(fold_csv_path).stem}_integrated.csv"))
    
    result_df.to_csv(output_path, index=False)
    
    print(f"匹配完成！")
    print(f"原始数据量: fold_df({len(fold_df)}行), nifti_df({len(nifti_df)}行)")
    print(f"成功匹配并提取的数据量: {len(result_df)}行")
    print(f"结果已保存至: {output_path}")
    
    return result_df

# 主函数部分保持不变...
if __name__ == "__main__":
    nifti_path = "/data/nifti_image_paths.csv"
    fold_path = "/data/yolo_dataset_249/Kfold/fold_assignment.csv"
    
    if os.path.exists(nifti_path) and os.path.exists(fold_path):
        integrate_csv_files(nifti_path, fold_path)
    else:
        print("请检查文件路径是否存在。")