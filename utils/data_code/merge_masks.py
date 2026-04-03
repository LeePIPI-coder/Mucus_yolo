import os
import nibabel as nib
import numpy as np


"""
这个脚本用于合并每个病人目录下的airway_mask.nii.gz和mask.nii.gz文件，生成一个新的combined_mask.nii.gz文件
其中：
- airway_mask.nii.gz中的非零像素值被设置为1，表示气道区域；
- mask.nii.gz中的非零像素值被设置为2，表示非气道区域。
"""
# 数据集根目录
ROOT_DIR = '/data/Mucus_data'

# 遍历所有病人目录
for patient_id in os.listdir(ROOT_DIR):
    if patient_id != 'E0001485_20111111':
        continue
    patient_dir = os.path.join(ROOT_DIR, patient_id)
    
    # 检查是否是目录
    if not os.path.isdir(patient_dir):
        continue
    
    print(f"Processing patient: {patient_id}")
    
    # 构建文件路径
    airway_mask_path = os.path.join(patient_dir, 'airway_mask.nii.gz')
    mask_path = os.path.join(patient_dir, 'mask.nii.gz')
    output_path = os.path.join(patient_dir, 'combined_mask.nii.gz')
    
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
        
        # 创建合并后的掩码
        combined_data = np.zeros_like(airway_data, dtype=np.uint8)
        
        # 先设置气道掩码为1
        combined_data[airway_data > 0] = 1
        
        # 再设置mask掩码为2（覆盖重叠区域）
        combined_data[mask_data > 0] = 2
        
        # 保存合并后的掩码
        combined_img = nib.Nifti1Image(combined_data, airway_img.affine)
        nib.save(combined_img, output_path)
        
        print(f"  Combined mask saved to: {output_path}")
        
    except Exception as e:
        print(f"  Error processing patient {patient_id}: {str(e)}")

print("\nProcessing complete!")