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
    parser.add_argument('--dicom_folder', type=str, required=True, 
                        help='DICOM序列文件夹路径')
    parser.add_argument('--mask_path', type=str, required=True, 
                        help='3D掩码文件路径 (.nii 或 .nii.gz)')
    parser.add_argument('--output_dir', type=str, default='./hu_analysis_output',
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading DICOM series...")
    ct_volume = load_dicom_series(args.dicom_folder)
    print(f"CT Volume shape: {ct_volume.shape}")
    
    print("Loading mask...")
    mask = load_nifti_mask(args.mask_path)
    print(f"Mask shape: {mask.shape}")
    
    print("Calculating HU distribution...")
    values, hist, bin_centers, stats = calculate_hu_distribution(ct_volume, mask)
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(args.mask_path))[0]
    txt_output_path = os.path.join(args.output_dir, f"{base_name}_hu_distribution.txt")
    plot_output_path = os.path.join(args.output_dir, f"{base_name}_hu_distribution.png")
    
    print("Saving distribution to txt file...")
    save_distribution_to_txt(values, hist, bin_centers, stats, txt_output_path)
    
    print("Generating distribution plot...")
    plot_hu_distribution(values, hist, bin_centers, stats, plot_output_path)
    
    print(f"\nAnalysis completed!")
    print(f"Results saved to:")
    print(f"  Text file: {txt_output_path}")
    print(f"  Plot: {plot_output_path}")
    
    print(f"\nStatistical Summary:")
    print(f"  Mean HU: {stats['mean']:.2f}")
    print(f"  Std Dev: {stats['std']:.2f}")
    print(f"  Min HU: {stats['min']:.2f}")
    print(f"  Max HU: {stats['max']:.2f}")
    print(f"  Total voxels in mask: {stats['total_voxels']:,}")


if __name__ == "__main__":
    main()