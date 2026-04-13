import os
import pandas as pd
from pathlib import Path

def find_dicom_paths(csv_file_path, data_directory, target_folds=None):
    """
    根据CSV表格内容查找对应分折数据的DICOM目录地址
    
    Args:
        csv_file_path (str): CSV文件路径，包含val_fold和patient_key列
        data_directory (str): 数据根目录，用于搜索DICOM目录
        target_folds (list or int, optional): 指定要查找的折数，如果为None则返回所有数据
    
    Returns:
        dict: 包含patient_key到DICOM绝对路径的映射字典
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 如果指定了特定的折数，则只筛选这些数据
    if target_folds is not None:
        if isinstance(target_folds, int):
            target_folds = [target_folds]
        df = df[df['val_fold'].isin(target_folds)]
    
    # 存储结果的字典
    dicom_paths = {}
    
    # 获取数据目录中的所有子目录
    data_root = Path(data_directory)
    
    # 遍历每个patient_key
    for _, row in df.iterrows():
        patient_key = row['patient_key']
        val_fold = row['val_fold']
        
        # 在数据目录中查找匹配的patient_key
        found_path = None
        
        # 遍历数据目录下的所有子目录，寻找匹配的patient_key
        for dir_path in data_root.rglob('*'):
            if dir_path.is_dir() and patient_key in dir_path.name:
                found_path = str(dir_path.absolute())
                break
        
        if found_path:
            dicom_paths[patient_key] = {
                'path': found_path,
                'val_fold': val_fold
            }
            print(f"Found {patient_key} in fold {val_fold}: {found_path}")
        else:
            dicom_paths[patient_key] = {
                'path': None,
                'val_fold': val_fold
            }
            print(f"Not found {patient_key} in fold {val_fold}")
    
    return dicom_paths

def get_dicom_for_fold(csv_file_path, data_directory, fold_number):
    """
    获取指定折数的所有DICOM路径
    
    Args:
        csv_file_path (str): CSV文件路径
        data_directory (str): 数据根目录
        fold_number (int): 要获取的折数
    
    Returns:
        dict: 指定折数的patient_key到DICOM路径的映射
    """
    return find_dicom_paths(csv_file_path, data_directory, target_folds=fold_number)

def main(csv_file_path, data_directory):
    """
    根据分折数据表获取所有数据在原始目录中的DICOM目录路径并保存到CSV文件
    """
    
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"CSV文件不存在: {csv_file_path}")
        return
    
    if not os.path.exists(data_directory):
        print(f"数据目录不存在: {data_directory}")
        return
    
    print("开始查找DICOM目录...")
    
    # 查找所有数据的DICOM路径
    all_dicom_paths = find_dicom_paths(csv_file_path, data_directory)
    
    # 输出统计信息
    found_count = sum(1 for v in all_dicom_paths.values() if v['path'] is not None)
    total_count = len(all_dicom_paths)
    
    print(f"\n总计: {total_count} 个患者ID")
    print(f"找到: {found_count} 个DICOM目录")
    print(f"未找到: {total_count - found_count} 个DICOM目录")

    return all_dicom_paths

if __name__ == "__main__":
    
    # 设置文件路径
    csv_file_path = '/data/yolo_dataset_249/Kfold_neg_0/5_scan.csv'
    data_directory = '/data/Mucus_origin_data'
    results = main(csv_file_path, data_directory)
    # 将字典结果转换为适当的DataFrame格式
    rows = []
    for patient_key, info in results.items():
        rows.append({
            'patient_key': patient_key,
            'val_fold': info['val_fold'],
            'dicom_path': info['path']
        })
    
    root_df = pd.DataFrame(rows)
    root_df.to_csv('/data/yolo_dataset_249/Kfold_neg_0/5_scan_data.csv', index=False)
    print(f"\n结果已保存到 /data/yolo_dataset_249/Kfold_neg_0/5_scan_data.csv") 
    print(f"共保存了 {len(root_df)} 行数据")