"""
统计3D掩码对应位置CT图体素的HU值分布
输入：DICOM序列文件夹 和 对应的3D掩码.nii.gz文件
输出：HU值分布的txt文件 和 分布图
"""

import os
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import Counter
import cv2
import pandas as pd


def load_dicom_series(dicom_folder):
    """加载DICOM序列，返回3D体积数据"""
    files = [p for p in Path(dicom_folder).iterdir() if p.suffix.lower() in ['.dcm', '.dicom']]
    if not files:
        # 如果直接子目录没有找到，递归搜索
        files = list(Path(dicom_folder).rglob('*.dcm')) + list(Path(dicom_folder).rglob('*.dicom'))
    
    if not files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")
    
    dicoms = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p))
            dicoms.append((p, ds))
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}")
            continue
    
    if len(dicoms) == 0:
        raise ValueError(f"No valid DICOM files found in {dicom_folder}")
    
    # 按InstanceNumber排序
    def key_fn(item):
        ds = item[1]
        return getattr(ds, 'InstanceNumber', 0) or getattr(ds, 'SliceLocation', 0) or str(item[0])
    
    dicoms.sort(key=key_fn)
    slices = []
    
    for _, ds in dicoms:
        # 获取像素数据
        pixel_array = ds.pixel_array
        
        # 应用Rescale Intercept和Slope转换为HU值
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        slope = float(getattr(ds, 'RescaleSlope', 1))
        
        if slope != 1:
            pixel_array = pixel_array * slope
        pixel_array = pixel_array + intercept
        
        slices.append(pixel_array)
    
    # 堆叠成3D体积 (Z, H, W)
    vol = np.stack(slices, axis=0)
    return vol


def load_nifti_image(nifti_path):
    """加载NIfTI格式的CT图像"""
    img = nib.load(str(nifti_path))
    ct_data = img.get_fdata()
    return ct_data


def load_nifti_mask(nifti_path):
    """加载NIfTI格式的3D掩码"""
    img = nib.load(str(nifti_path))
    mask_data = img.get_fdata()
    
    # 确保掩码是二值化的 (0和非0)
    mask_data = (mask_data > 0).astype(np.uint8)
    
    return mask_data


def calculate_hu_distribution(ct_volume, mask):
    """计算掩码区域内CT体素的HU值分布"""
    # 确保掩码和CT体积尺寸匹配
    if ct_volume.shape != mask.shape:
        print(f"Warning: CT volume shape {ct_volume.shape} != mask shape {mask.shape}")
        # 尝试调整掩码大小以匹配CT体积
        if len(ct_volume.shape) == 3 and len(mask.shape) == 3:
            resized_mask = np.zeros_like(ct_volume, dtype=np.uint8)
            for i in range(ct_volume.shape[0]):
                target_slice = cv2.resize(
                    mask[i, :, :].astype(np.float32),
                    (ct_volume.shape[2], ct_volume.shape[1]),
                    interpolation=cv2.INTER_NEAREST
                )
                resized_mask[i, :, :] = (target_slice > 0).astype(np.uint8)
            mask = resized_mask
        else:
            raise ValueError(f"Cannot align shapes: {ct_volume.shape} and {mask.shape}")
    
    # 提取掩码区域内的HU值
    masked_values = ct_volume[mask > 0]
    
    if len(masked_values) == 0:
        raise ValueError("Mask contains no foreground pixels")
    
    # 统计分布
    hist, bin_edges = np.histogram(masked_values, bins=200, range=(-1000, 2000))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 计算统计信息
    stats = {
        'mean': float(np.mean(masked_values)),
        'std': float(np.std(masked_values)),
        'min': float(np.min(masked_values)),
        'max': float(np.max(masked_values)),
        'median': float(np.median(masked_values)),
        'q25': float(np.percentile(masked_values, 25)),
        'q75': float(np.percentile(masked_values, 75)),
        'total_voxels': int(len(masked_values))
    }
    
    return masked_values, hist, bin_centers, stats


def save_distribution_to_txt(values, hist, bin_centers, stats, output_txt_path):
    """将HU值分布保存到txt文件"""
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("HU Value Distribution Analysis\n")
        f.write("="*40 + "\n\n")
        
        f.write("Statistical Summary:\n")
        f.write("-"*20 + "\n")
        f.write(f"Mean HU: {stats['mean']:.2f}\n")
        f.write(f"Std Deviation: {stats['std']:.2f}\n")
        f.write(f"Min HU: {stats['min']:.2f}\n")
        f.write(f"Max HU: {stats['max']:.2f}\n")
        f.write(f"Median HU: {stats['median']:.2f}\n")
        f.write(f"25th Percentile: {stats['q25']:.2f}\n")
        f.write(f"75th Percentile: {stats['q75']:.2f}\n")
        f.write(f"Total Voxels in Mask: {stats['total_voxels']:,}\n\n")
        
        f.write("HU Value Distribution:\n")
        f.write("-"*20 + "\n")
        f.write("HU Range\tCount\tPercentage\n")
        
        total_count = np.sum(hist)
        for i in range(len(hist)):
            if hist[i] > 0:  # 只输出有值的区间
                range_start = bin_centers[i] - (bin_centers[1] - bin_centers[0])/2
                range_end = bin_centers[i] + (bin_centers[1] - bin_centers[0])/2
                percentage = (hist[i] / total_count) * 100
                f.write(f"[{range_start:.1f}, {range_end:.1f}]\t{hist[i]}\t{percentage:.2f}%\n")


def save_summary_to_txt(all_stats, all_values, output_txt_path):
    """将所有案例的HU值分布保存到一个txt文件"""
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("HU Value Distribution Analysis - All Cases\n")
        f.write("="*60 + "\n\n")
        
        # 写入每个案例的统计信息
        f.write("Individual Case Statistics:\n")
        f.write("-"*60 + "\n")
        f.write("Case\tMean\tStd\tMin\tMax\tMedian\tQ25\tQ75\tVoxels\n")
        
        for case_name, stats in all_stats.items():
            f.write(f"{case_name}\t{stats['mean']:.2f}\t{stats['std']:.2f}\t{stats['min']:.2f}\t{stats['max']:.2f}\t{stats['median']:.2f}\t{stats['q25']:.2f}\t{stats['q75']:.2f}\t{stats['total_voxels']:,}\n")
        
        # 计算整体统计信息
        all_hu_values = np.concatenate(list(all_values.values()))
        overall_stats = {
            'mean': float(np.mean(all_hu_values)),
            'std': float(np.std(all_hu_values)),
            'min': float(np.min(all_hu_values)),
            'max': float(np.max(all_hu_values)),
            'median': float(np.median(all_hu_values)),
            'q25': float(np.percentile(all_hu_values, 25)),
            'q75': float(np.percentile(all_hu_values, 75)),
            'total_voxels': int(len(all_hu_values))
        }
        
        f.write("\nOverall Statistics:\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean HU: {overall_stats['mean']:.2f}\n")
        f.write(f"Std Deviation: {overall_stats['std']:.2f}\n")
        f.write(f"Min HU: {overall_stats['min']:.2f}\n")
        f.write(f"Max HU: {overall_stats['max']:.2f}\n")
        f.write(f"Median HU: {overall_stats['median']:.2f}\n")
        f.write(f"25th Percentile: {overall_stats['q25']:.2f}\n")
        f.write(f"75th Percentile: {overall_stats['q75']:.2f}\n")
        f.write(f"Total Voxels in All Masks: {overall_stats['total_voxels']:,}\n")


def plot_summary_distribution(all_values, output_plot_path):
    """绘制所有案例的HU值分布到一个表中"""
    # 合并所有HU值
    all_hu_values = np.concatenate(list(all_values.values()))
    
    # 计算整体分布
    hist, bin_edges = np.histogram(all_hu_values, bins=200, range=(-1000, 2000))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 计算整体统计信息
    overall_stats = {
        'mean': float(np.mean(all_hu_values)),
        'std': float(np.std(all_hu_values)),
        'min': float(np.min(all_hu_values)),
        'max': float(np.max(all_hu_values)),
        'median': float(np.median(all_hu_values)),
        'q25': float(np.percentile(all_hu_values, 25)),
        'q75': float(np.percentile(all_hu_values, 75)),
        'total_voxels': int(len(all_hu_values))
    }
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('HU Value Distribution - All Cases', fontsize=16)
    
    # 直方图
    axes[0, 0].bar(bin_centers, hist, width=np.diff(bin_centers)[0]*0.8, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('HU Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('HU Value Histogram (All Cases)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加统计线
    axes[0, 0].axvline(overall_stats['mean'], color='red', linestyle='--', label=f'Mean: {overall_stats["mean"]:.2f}')
    axes[0, 0].axvline(overall_stats['median'], color='orange', linestyle='--', label=f'Median: {overall_stats["median"]:.2f}')
    axes[0, 0].legend()
    
    # 密度图
    axes[0, 1].plot(bin_centers, hist / np.sum(hist), linewidth=2, color='green')
    axes[0, 1].set_xlabel('HU Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('HU Value Density Plot (All Cases)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 箱线图
    box_data = [all_hu_values]
    axes[1, 0].boxplot(box_data, labels=['HU Values (All Cases)'])
    axes[1, 0].set_ylabel('HU Value')
    axes[1, 0].set_title('Box Plot of HU Values (All Cases)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 统计信息表
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = [
        ['Statistic', 'Value'],
        ['Mean', f'{overall_stats["mean"]:.2f}'],
        ['Std Dev', f'{overall_stats["std"]:.2f}'],
        ['Min', f'{overall_stats["min"]:.2f}'],
        ['Max', f'{overall_stats["max"]:.2f}'],
        ['Median', f'{overall_stats["median"]:.2f}'],
        ['Q25', f'{overall_stats["q25"]:.2f}'],
        ['Q75', f'{overall_stats["q75"]:.2f}'],
        ['Voxel Count', f'{overall_stats["total_voxels"]:,}']
    ]
    
    table = axes[1, 1].table(cellText=table_data,
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    table[(0, 0)].set_facecolor('#4CAF50')
    table[(0, 1)].set_facecolor('#4CAF50')
    table[(0, 0)].set_text_props(weight='bold', color='white')
    table[(0, 1)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Overall Statistical Summary')
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_hu_distribution(values, hist, bin_centers, stats, output_plot_path):
    """绘制HU值分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('HU Value Distribution in Masked Region', fontsize=16)
    
    # 直方图
    axes[0, 0].bar(bin_centers, hist, width=np.diff(bin_centers)[0]*0.8, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('HU Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('HU Value Histogram')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加统计线
    axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.2f}')
    axes[0, 0].axvline(stats['median'], color='orange', linestyle='--', label=f'Median: {stats["median"]:.2f}')
    axes[0, 0].legend()
    
    # 密度图
    axes[0, 1].plot(bin_centers, hist / np.sum(hist), linewidth=2, color='green')
    axes[0, 1].set_xlabel('HU Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('HU Value Density Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 箱线图
    box_data = [values]
    axes[1, 0].boxplot(box_data, labels=['HU Values'])
    axes[1, 0].set_ylabel('HU Value')
    axes[1, 0].set_title('Box Plot of HU Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 统计信息表
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = [
        ['Statistic', 'Value'],
        ['Mean', f'{stats["mean"]:.2f}'],
        ['Std Dev', f'{stats["std"]:.2f}'],
        ['Min', f'{stats["min"]:.2f}'],
        ['Max', f'{stats["max"]:.2f}'],
        ['Median', f'{stats["median"]:.2f}'],
        ['Q25', f'{stats["q25"]:.2f}'],
        ['Q75', f'{stats["q75"]:.2f}'],
        ['Voxel Count', f'{stats["total_voxels"]:,}']
    ]
    
    table = axes[1, 1].table(cellText=table_data,
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    table[(0, 0)].set_facecolor('#4CAF50')
    table[(0, 1)].set_facecolor('#4CAF50')
    table[(0, 0)].set_text_props(weight='bold', color='white')
    table[(0, 1)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Statistical Summary')
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='统计3D掩码对应位置CT图体素的HU值分布')
    parser.add_argument('--csv_path', type=str, default='/data/nifti_image_paths.csv', 
                        help='CSV文件路径，包含CT图像路径')
    parser.add_argument('--output_dir', type=str, default='./hu_analysis_output',
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取CSV文件
    print(f"Reading CSV file: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    if 'path' not in df.columns:
        raise ValueError("CSV文件必须包含'path'列")
    
    # 收集所有案例的统计信息和HU值
    all_stats = {}
    all_values = {}
    
    # 处理每个CT图像路径
    for idx, row in df.iterrows():
        ct_path = row['path']
        
        ## ----------------------
        if idx >=1 :
            continue
        ct_path = "/data/nifti_files/image/E0001018_20091229.nii.gz"
        
        # 生成对应的mask路径
        mask_path = ct_path.replace('image', 'mask')
        
        print(f"\nProcessing {idx+1}/{len(df)}: {ct_path}")
        print(f"Corresponding mask: {mask_path}")
        
        # 检查文件是否存在
        if not os.path.exists(ct_path):
            print(f"Warning: CT file not found: {ct_path}")
            continue
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found: {mask_path}")
            continue
        
        print("Loading CT image...")
        ct_volume = load_nifti_image(ct_path)
        ct_volume = ct_volume.transpose(1, 0, 2)
        print(f"CT Volume shape: {ct_volume.shape}")
        
        print("Loading mask...")
        mask = load_nifti_mask(mask_path)
        print(f"Mask shape: {mask.shape}")
        
        print("Calculating HU distribution...")
        values, hist, bin_centers, stats = calculate_hu_distribution(ct_volume, mask)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(ct_path))[0]
        txt_output_path = os.path.join(args.output_dir, f"{base_name}_hu_distribution1.txt")
        plot_output_path = os.path.join(args.output_dir, f"{base_name}_hu_distribution1.png")
        
        print(f"Saving distribution to txt file for {base_name}...")
        save_distribution_to_txt(values, hist, bin_centers, stats, txt_output_path)
        
        print(f"Generating distribution plot for {base_name}...")
        plot_hu_distribution(values, hist, bin_centers, stats, plot_output_path)
        
        print(f"\nAnalysis completed for {base_name}!")
        print(f"Results saved to:")
        print(f"  Text file: {txt_output_path}")
        print(f"  Plot: {plot_output_path}")
        
        print(f"\nStatistical Summary:")
        print(f"  Mean HU: {stats['mean']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f}")
        print(f"  Min HU: {stats['min']:.2f}")
        print(f"  Max HU: {stats['max']:.2f}")
        print(f"  Total voxels in mask: {stats['total_voxels']:,}")
        print("="*60)
        
        # 收集统计信息和HU值
        all_stats[base_name] = stats
        all_values[base_name] = values
    
    # 保存汇总统计到一个txt文件
    summary_txt_path = os.path.join(args.output_dir, "all_cases_hu_distribution_summary.txt")
    print(f"\nSaving summary of all cases to: {summary_txt_path}")
    save_summary_to_txt(all_stats, all_values, summary_txt_path)
    
    # 绘制汇总分布图
    summary_plot_path = os.path.join(args.output_dir, "all_cases_hu_distribution_summary.png")
    print(f"Generating summary distribution plot: {summary_plot_path}")
    plot_summary_distribution(all_values, summary_plot_path)
    
    print("\nAll cases processed!")
    print(f"Summary results saved to:")
    print(f"  Summary text file: {summary_txt_path}")
    print(f"  Summary plot: {summary_plot_path}")


if __name__ == "__main__":
    main()