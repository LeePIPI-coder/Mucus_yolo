import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# 确保输出目录存在
def ensure_output_dir():
    output_dir = '/workspace/hu_analysis_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# 读取NIfTI文件
def process_nii_file(nii_path):
    # 加载NIfTI文件
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    # 获取Z轴中间层
    z_dim = data.shape[2]
    middle_slice_idx = z_dim // 2
    middle_slice = data[:, :, middle_slice_idx]
    
    # 创建红色掩码（HU值在[0,86]之间）
    red_mask = np.zeros_like(middle_slice, dtype=np.uint8)
    red_mask[(middle_slice >= 0) & (middle_slice <= 86)] = 255
    
    # 创建蓝色掩码（HU值在[-224,-78]之间）
    blue_mask = np.zeros_like(middle_slice, dtype=np.uint8)
    blue_mask[(middle_slice >= -224) & (middle_slice <= -78)] = 255
    
    # 确保输出目录存在
    output_dir = ensure_output_dir()
    
    # 保存红色掩码图像
    plt.imsave(f'{output_dir}/red_mask.png', red_mask, cmap='Reds')
    print(f"红色掩码已保存到 {output_dir}/red_mask.png")
    
    # 保存蓝色掩码图像
    plt.imsave(f'{output_dir}/blue_mask.png', blue_mask, cmap='Blues')
    print(f"蓝色掩码已保存到 {output_dir}/blue_mask.png")
    
    # 将掩码覆盖在CT原图上
    overlay_with_masks(middle_slice, red_mask, blue_mask, output_dir)
    
    return middle_slice, red_mask, blue_mask

def overlay_with_masks(ct_slice, red_mask, blue_mask, output_dir):
    """将红色和蓝色掩码覆盖在CT原图上"""
    # 归一化CT图像到0-255范围
    ct_norm = ((ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min()) * 255).astype(np.uint8)
    
    # 创建RGB图像
    rgb_image = np.stack((ct_norm, ct_norm, ct_norm), axis=-1)
    
    # 应用红色掩码
    rgb_image[red_mask == 255, 0] = 255  # 红色通道设为255
    rgb_image[red_mask == 255, 1] = 0    # 绿色通道设为0
    rgb_image[red_mask == 255, 2] = 0    # 蓝色通道设为0
    
    # 应用蓝色掩码
    rgb_image[blue_mask == 255, 0] = 0    # 红色通道设为0
    rgb_image[blue_mask == 255, 1] = 0    # 绿色通道设为0
    rgb_image[blue_mask == 255, 2] = 255  # 蓝色通道设为255
    
    # 保存覆盖后的图像
    plt.imsave(f'{output_dir}/ct_with_masks.png', rgb_image)
    print(f"带掩码的CT图像已保存到 {output_dir}/ct_with_masks.png")

 
# 主函数
if __name__ == "__main__":
    """
    根据HU值范围提取CT图像中的组织类型，并将结果保存为PNG图像。
    """
    ct_path = "/data/nifti_files/image/E0001018_20091229.nii.gz"
    middle_slice, red_mask, blue_mask = process_nii_file(ct_path)
    print(f"处理完成！中间层索引: {middle_slice.shape}")
    print(f"红色掩码中非零像素数: {np.sum(red_mask > 0)}")
    print(f"蓝色掩码中非零像素数: {np.sum(blue_mask > 0)}")