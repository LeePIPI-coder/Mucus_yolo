import os
import argparse
import numpy as np
import nibabel as nib
from gzip import decompress
from pathlib import Path
import json
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def decode_mask_Byte2Bit(mask_bytes: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    """
    将压缩前的 bit-打包字节流还原为 mask_array
    shape: (z, x, y)
    """
    z, x, y = shape
    arr = np.zeros(shape, dtype=np.uint8)

    row_bits_len = x
    row_bytes_len = (x + 7) // 8

    buf = memoryview(mask_bytes)
    offset = 0
    for zi in range(z):
        for yi in range(y):
            row = np.unpackbits(
                np.frombuffer(buf[offset:offset+row_bytes_len], dtype=np.uint8)
            )[:row_bits_len]   # 截断到 x 长度
            arr[zi, :, yi] = row
            offset += row_bytes_len
    return arr

def get_dimensions_from_filename(filename: str):
    """
    从文件名中提取维度信息
    文件名格式: lesionAnnot3D-000.512x512x100.psegmaskz
    """
    try:
        # 提取维度部分
        dim_part = filename.split('.')[-2]  # 获取 "512x512x100"
        dimensions = dim_part.split('x')    # 分割成 ['512', '512', '100']
        if len(dimensions) == 3:
            return tuple(map(int, dimensions))
        else:
            return None
    except:
        return None

def read_psegmaskz_file(file_path: str):
    """
    读取psegmaskz文件并解码为numpy数组
    """
    try:
        # 读取gzip压缩的数据
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        
        # 解压缩
        decompressed_data = decompress(compressed_data)
        
        # 从文件名获取维度信息
        filename = Path(file_path).name
        dimensions = get_dimensions_from_filename(filename)
        
        if dimensions is None:
            # 如果无法从文件名获取维度，尝试从JSON文件获取
            json_file = Path(file_path).parent / "lesionAnnot3D.json"
            if json_file.exists():
                dimensions = get_dimensions_from_json(json_file, filename)
        
        if dimensions is None:
            raise ValueError(f"无法从文件名 {filename} 或JSON文件获取维度信息")
        
        # 解码为numpy数组
        mask_array = decode_mask_Byte2Bit(decompressed_data, dimensions)
        
        return mask_array, dimensions
    except Exception as e:
        print(f"读取psegmaskz文件失败: {e}")
        raise e

if __name__ == "__main__":
    
    segmaskz_file = r"S:\AVIEWDB\mucus_plug\dcm\2009\11\ST_40351_00000151\SE_00002_00000151\stor\results\lesionAnnot3D-000.512x512x547.psegmaskz"
    mask_array, dimensions = read_psegmaskz_file(segmaskz_file)
    
    print(f"segmaskz解码后数组形状: {mask_array.shape}")
    print(f"文件名中的维度: {dimensions}")
    
    # 尝试加载对应的 nii.gz 文件进行比较
    # 构造对应的 nii.gz 文件路径
    nii_gz_path = segmaskz_file.replace('.512x512x547.psegmaskz', '.nii.gz')
    
    print(f"尝试加载对应的 NIfTI 文件: {nii_gz_path}")
    
    try:
        # 加载 nii.gz 文件
        nii_img = nib.load(nii_gz_path)
        nii_data = nii_img.get_fdata()
        print(f"NIfTI 文件数组形状: {nii_data.shape}")
        
        # 比较两个数组的形状
        print(f"\n形状比较:")
        print(f"segmaskz数组: {mask_array.shape}")
        print(f"nii.gz数组:   {nii_data.shape}")
        
        # 检查非零元素数量
        segmaskz_nonzeros = np.count_nonzero(mask_array)
        niigz_nonzeros = np.count_nonzero(nii_data)
        print(f"\n非零元素数量:")
        print(f"segmaskz数组: {segmaskz_nonzeros}")
        print(f"nii.gz数组:   {niigz_nonzeros}")
        
        # 找出两个数组中非零元素的位置
        segmaskz_indices = np.where(mask_array > 0)
        niigz_indices = np.where(nii_data > 0)
        
        print(f"\nsegmaskz数组中前10个非零元素的位置:")
        for i in range(min(10, len(segmaskz_indices[0]))):
            idx0, idx1, idx2 = segmaskz_indices[0][i], segmaskz_indices[1][i], segmaskz_indices[2][i]
            print(f"({idx0}, {idx1}, {idx2})")
        
        print(f"\nnii.gz数组中前10个非零元素的位置:")
        for i in range(min(10, len(niigz_indices[0]))):
            idx0, idx1, idx2 = niigz_indices[0][i], niigz_indices[1][i], niigz_indices[2][i]
            print(f"({idx0}, {idx1}, {idx2})")
        
        # 尝试确定坐标轴对应关系
        print(f"\n坐标轴对应关系分析:")
        if mask_array.shape == nii_data.shape:
            print(f"两个数组形状相同 ({mask_array.shape})，坐标轴顺序可能一致")
        else:
            print(f"两个数组形状不同，需要进一步分析坐标轴对应关系")
            
            # 检查是否是轴的转置关系
            if mask_array.shape == nii_data.shape[::-1]:
                print(f"segmaskz数组形状是nii.gz数组形状的反转")
            elif mask_array.shape == tuple(reversed(nii_data.shape)):
                print(f"segmaskz数组形状与nii.gz数组形状互为反向")
            else:
                print(f"两数组形状无明显转置关系")
        
        # 找出所有大于0的值的空间位置
        indices = np.where(mask_array > 0)
        print(f"\nsegmaskz解码数组中找到 {len(indices[0])} 个大于0的值")
        
        # 基于形状分析确定坐标轴顺序
        if mask_array.shape == dimensions:
            # 数组形状与维度信息一致，即 (z, x, y)
            print("segmaskz数组中大于0的值的空间位置 (z, x, y):")
            for i in range(min(10, len(indices[0]))):  # 先只输出前10个作为示例
                z, x, y = indices[0][i], indices[1][i], indices[2][i]
                print(f"({z}, {x}, {y})")
            if len(indices[0]) > 10:
                print(f"... 还有 {len(indices[0]) - 10} 个点")
        elif mask_array.shape[::-1] == dimensions:
            # 数组形状是维度信息的反转，即 (z, y, x) 或 (x, y, z)
            print("segmaskz数组中大于0的值的空间位置 (x, y, z):")
            for i in range(min(10, len(indices[0]))):  # 先只输出前10个作为示例
                x, y, z = indices[0][i], indices[1][i], indices[2][i]
                print(f"({x}, {y}, {z})")
            if len(indices[0]) > 10:
                print(f"... 还有 {len(indices[0]) - 10} 个点")
        else:
            print(f"segmaskz数组形状 {mask_array.shape} 与预期维度 {dimensions} 不匹配")
            print("按实际索引顺序输出 (axis0, axis1, axis2):")
            for i in range(min(10, len(indices[0]))):  # 先只输出前10个作为示例
                idx0, idx1, idx2 = indices[0][i], indices[1][i], indices[2][i]
                print(f"({idx0}, {idx1}, {idx2})")
            if len(indices[0]) > 10:
                print(f"... 还有 {len(indices[0]) - 10} 个点")
                
    except FileNotFoundError:
        print(f"未找到对应的 NIfTI 文件: {nii_gz_path}")
        print(f"正在继续分析 segmaskz 解码数组...")
        
        # 找出所有大于0的值的空间位置
        indices = np.where(mask_array > 0)
        print(f"找到 {len(indices[0])} 个大于0的值")
        
        if mask_array.shape == dimensions:
            # 数组形状与维度信息一致，即 (z, x, y)
            print("所有大于0的值的空间位置 (z, x, y):")
            for i in range(min(10, len(indices[0]))):  # 先只输出前10个作为示例
                z, x, y = indices[0][i], indices[1][i], indices[2][i]
                print(f"({z}, {x}, {y})")
            if len(indices[0]) > 10:
                print(f"... 还有 {len(indices[0]) - 10} 个点")
        elif mask_array.shape[::-1] == dimensions:
            # 数组形状是维度信息的反转，即 (z, y, x) 或 (x, y, z)
            print("所有大于0的值的空间位置 (x, y, z):")
            for i in range(min(10, len(indices[0]))):  # 先只输出前10个作为示例
                x, y, z = indices[0][i], indices[1][i], indices[2][i]
                print(f"({x}, {y}, {z})")
            if len(indices[0]) > 10:
                print(f"... 还有 {len(indices[0]) - 10} 个点")
        else:
            print(f"实际数组形状 {mask_array.shape} 与预期维度 {dimensions} 不匹配")
            print("按实际索引顺序输出 (axis0, axis1, axis2):")
            for i in range(min(10, len(indices[0]))):  # 先只输出前10个作为示例
                idx0, idx1, idx2 = indices[0][i], indices[1][i], indices[2][i]
                print(f"({idx0}, {idx1}, {idx2})")
            if len(indices[0]) > 10:
                print(f"... 还有 {len(indices[0]) - 10} 个点")