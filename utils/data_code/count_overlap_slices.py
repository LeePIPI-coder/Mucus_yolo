import os
import nibabel as nib
import numpy as np

# 数据集根目录
ROOT_DIR = '/data/Mucus_data'

# 初始化计数器
overlap_pixels = 0
mask_pixels = 0

overlap_slices = 0
mask_slices = 0

# 遍历所有病人目录
for patient_id in os.listdir(ROOT_DIR):
    patient_dir = os.path.join(ROOT_DIR, patient_id)
    
    # 检查是否是目录
    if not os.path.isdir(patient_dir):
        continue
    
    print(f"Processing patient: {patient_id}")
    
    # 构建文件路径
    airway_mask_path = os.path.join(patient_dir, 'airway_mask.nii.gz')
    mask_path = os.path.join(patient_dir, 'mask.nii.gz')
    
    # 检查文件是否存在
    if not os.path.exists(airway_mask_path):
        print(f"  Missing airway_mask.nii.gz for patient {patient_id}")
        continue
    
    if not os.path.exists(mask_path):
        print(f"  Missing mask.nii.gz for patient {patient_id}")
        continue
    
    # 加载掩码文件
    try:
        airway_img = nib.load(airway_mask_path)
        airway_data = airway_img.get_fdata()
        
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        
        # 确保两个掩码形状一致
        if airway_data.shape != mask_data.shape:
            print(f"  Mismatched shapes for patient {patient_id}: airway={airway_data.shape}, mask={mask_data.shape}")
            continue
        
        # 获取切片数量（z轴）
        num_slices = airway_data.shape[2]
        
        # 计算每个切片的像素数
        slice_pixels = airway_data.shape[0] * airway_data.shape[1]
        
        # 统计重合像素数和包含粘液栓的像素数
        patient_overlap_pixels = 0
        patient_mask_pixels = 0
        patient_overlap_slices = 0
        patient_mask_slices = 0
        
        
        for z in range(num_slices):
            airway_slice = airway_data[:, :, z]
            mask_slice = mask_data[:, :, z]
            
            # 创建布尔数组
            airway_bool = airway_slice > 0
            mask_bool = mask_slice > 0
            
            # 计算重合像素数
            overlap_bool = airway_bool & mask_bool
            slice_overlap_pixels = np.sum(overlap_bool)
            # 检查是否存在重合
            has_overlap = slice_overlap_pixels > 0
            
            # 检查是否存在相邻像素
            has_adjacent = False
            if not has_overlap and np.any(airway_bool) and np.any(mask_bool):
                # 对气道掩码进行膨胀
                airway_dilated = np.zeros_like(airway_bool)
                for i in range(airway_bool.shape[0]):
                    for j in range(airway_bool.shape[1]):
                        if airway_bool[i, j]:
                            # 处理边界情况
                            start_i = max(0, i-1)
                            end_i = min(airway_bool.shape[0], i+2)
                            start_j = max(0, j-1)
                            end_j = min(airway_bool.shape[1], j+2)
                            airway_dilated[start_i:end_i, start_j:end_j] = True
                # 检查膨胀后的气道掩码与粘液栓掩码是否有重叠
                has_adjacent = np.any(airway_dilated & mask_bool)
            
            if has_overlap or has_adjacent:
                patient_overlap_slices += 1
            patient_overlap_pixels += slice_overlap_pixels
            
            # 计算粘液栓像素数
            slice_mask_pixels = np.sum(mask_bool)
            if slice_mask_pixels > 0:
                patient_mask_slices += 1
            patient_mask_pixels += slice_mask_pixels
        
        # 更新总计数
        overlap_pixels += patient_overlap_pixels
        mask_pixels += patient_mask_pixels
        overlap_slices += patient_overlap_slices
        mask_slices += patient_mask_slices
        # 计算该病人的重合像素占比
        patient_overlap_ratio = (patient_overlap_pixels / patient_mask_pixels * 100) if patient_mask_pixels > 0 else 0
        patient_overlap_slice_ratio = (patient_overlap_slices / patient_mask_slices * 100) if patient_mask_slices > 0 else 0
        print(f"  Patient {patient_id}: {patient_overlap_pixels} overlap pixels, {patient_mask_pixels} mask pixels, overlap ratio: {patient_overlap_ratio:.2f}%")
        print(f"  Patient {patient_id}: {patient_overlap_slices} overlap slices, {patient_mask_slices} mask slices")
    except Exception as e:
        print(f"  Error processing patient {patient_id}: {str(e)}")

# 计算重合像素占粘液栓像素的百分比
overlap_to_mask_ratio = (overlap_pixels / mask_pixels * 100) if mask_pixels > 0 else 0
# 计算切片级别比例
overlap_slice_ratio = (overlap_slices / mask_slices * 100) if mask_slices > 0 else 0

# 输出结果
print("\n=== Statistics ===")
print(f"Overlap slices: {overlap_slices}")
print(f"Mask slices: {mask_slices}")
print(f"Overlap pixels: {overlap_pixels}")
print(f"Mucus plug pixels: {mask_pixels}")
print(f"Overlap slice ratio: {overlap_slice_ratio:.2f}%")
print(f"Overlap pixel ratio (vs mucus plug): {overlap_to_mask_ratio:.2f}%")
print("==================")