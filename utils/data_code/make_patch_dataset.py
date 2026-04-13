"""
这个脚本用于从医学图像数据创建patch数据集。
程序会读取一个csv文件，文件中保存了各个niigz数据地址。
加载NIfTI图像和对应的掩码文件，应用HU窗口化，提取2.5D patch，
并将数据集分割为训练、验证和测试集。patch以.png格式保存图像，
掩码以.png格式保存（如果存在）。
"""

import array
import os
import SimpleITK as sitk
import numpy as np
import cv2
import pandas as pd
from collections import defaultdict
import argparse
from tqdm import tqdm
import time
import gc


def make_patch_data(args):
    
    # 从 CSV 读取路径，按比例划分 train/valid/test（75% / 10% / 15%）
    df = pd.read_csv(args.csv_path)
    # 支持列名为 path，或第一列即为路径
    path_col = "path" if "path" in df.columns else df.columns[0]
    paths = df[path_col].dropna().astype(str).tolist()
    # 创建保存目录
    savePath = str(args.save_path)
    if not os.path.isdir(savePath):
        os.makedirs(savePath)
        
    if args.split:
        n = len(paths)
        train_len = int(n * 0.75)
        # valid_len = train_len + int(n * 0.1)

        trainset = paths[0:train_len]
        validset = paths[train_len:n]
        # testset = paths[valid_len:n]

        print(f"total: {n}")
        print(f"set - train: {len(trainset)}, valid: {len(validset)}")
        saveSetName = ['train', 'valid']
        
        # 配置参数
        patch_size = 128  # patch的大小
        stride_hw = 64    # 滑动窗口的步长
        depth_channel = 3  # 2.5D的深度通道数

        # 创建一个新的字典来存储新的DataFrame,使用defaultdict(list)可以避免KeyError
        # new_df = defaultdict(list)

        # 遍历每个数据集（train, valid）；dataset 中为图像完整路径
        for i, dataset in enumerate([trainset, validset]):
            img_save_dir = os.path.join(savePath, saveSetName[i], 'images')
            mask_save_dir = os.path.join(savePath, saveSetName[i], 'masks')
            os.makedirs(img_save_dir, exist_ok=True)
            os.makedirs(mask_save_dir, exist_ok=True)

            for f, image_path in enumerate(dataset):
                # if f < 56:
                #     continue
                file = os.path.basename(image_path)
                print('-'*20 + f"processing {saveSetName[i]}--{file.replace('.nii.gz','')}--{f}/{len(dataset)}" + '-'*20)
                # new_df['filename'].append(file)
                # new_df['set_type'].append(saveSetName[i])

                # 掩码路径：同目录下 image -> mask，文件名不变
                img_dir = os.path.dirname(image_path)
                mask_dir = img_dir.replace('image', 'mask')
                mask_path = os.path.join(mask_dir, file)

                header = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(header)
                mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)) if os.path.isfile(mask_path) else None
                spacing = header.GetSpacing()
                z_spacing = spacing[2]
                
                # 计算提取的通道
                betweenGapSlice = int(5 / z_spacing)
                depth, height, width = image.shape
                extract_ch = [ch for ch in range(0, depth, betweenGapSlice)]

                # 如果有掩码，添加有掩码的通道
                if os.path.isfile(mask_path) and mask is not None:
                    for ch in range(0, depth):
                        try:
                            if np.sum(mask[ch,:,:]) > 0:
                                extract_ch.append(ch)
                        except Exception as e:
                            print(f"⚠️ 捕获到错误: {e}")
                            continue
                else:
                    mask = None
                
                # 遍历提取的通道
                for ch in tqdm(extract_ch, desc=f"processing channels"):
                    # 创建2.5D图像（当前通道及其相邻通道）
                    tiff_image = np.zeros((height, width, depth_channel), dtype=np.float32)
                    # if ch == 0, then extract the first 3 channels
                    if ch == 0:
                        re_ch = [0, 1, 2]
                        for z, idx in enumerate(re_ch):
                            tiff_image[:, :, z] = image[idx, :, :]
                    elif ch == depth-1:
                        re_ch = [depth-3, depth-2, depth-1]
                        for z, idx in enumerate(re_ch):
                            tiff_image[:, :, z] = image[idx, :, :]
                    else:
                        for z, idx in enumerate(range(ch-1, ch+2)):
                            idx = np.clip(idx, 0, depth-1)
                            tiff_image[:, :, z] = image[idx, :, :]

                    # HU窗口化（Hounsfield Units）
                    np.clip(tiff_image, -1000, 400, out=tiff_image)
                    # 归一化到0-255
                    tiff_image = ((tiff_image + 1000) / 1400 * 255).astype('uint8')

                    # 使用滑动窗口提取patch
                    for y in range(0, height - patch_size + 1, stride_hw):
                        for x in range(0, width - patch_size + 1, stride_hw):
                            # 处理掩码
                            img_name = f"{file.replace('.nii.gz','')}_slice{ch}_y{y}_x{x}.png"

                            if mask is not None:
                                patch_mask = mask[ch, y:y+patch_size, x:x+patch_size]
                                # 如果patch中包含目标（掩码值大于0），则保存
                                if np.sum(patch_mask) > 0:
                                    patch_mask[patch_mask > 0] = 255
                                    cv2.imwrite(os.path.join(mask_save_dir, img_name), np.expand_dims(patch_mask, -1))
                                    
                            # 保存patch图像
                            patch_img = tiff_image[y:y+patch_size, x:x+patch_size, :]
                            # np.save(os.path.join(img_save_dir, img_name.replace('.png', '.npy')), patch_img)
                            cv2.imwrite(os.path.join(img_save_dir, img_name), patch_img)
                del image
                del mask
                del header
                del tiff_image
                # 显式触发垃圾回收（可选）
                gc.collect()
                os.sync()
                time.sleep(0.5)
            
    if not args.split:
        n = len(paths)
        print(f"total: {n}")
        patch_size = 128  # patch的大小
        stride_hw = 64    # 滑动窗口的步长
        depth_channel = 3  # 2.5D的深度通道数

        # 创建一个新的字典来存储新的DataFrame,使用defaultdict(list)可以避免KeyError
        # new_df = defaultdict(list)

        # 遍历每个数据集（train, valid, test）；dataset 中为图像完整路径
        img_save_dir = os.path.join(savePath, 'All', 'images')
        mask_save_dir = os.path.join(savePath, 'All', 'masks')
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)

        for f, image_path in enumerate(paths):
                
            file = os.path.basename(image_path)
            print('-'*20 + f"processing All--{file.replace('.nii.gz','')}--{f+1}/{n}" + '-'*20)
            # new_df['filename'].append(file)
            # new_df['set_type'].append(saveSetName[i])

            # 掩码路径：同目录下 image -> mask，文件名不变
            img_dir = os.path.dirname(image_path)
            mask_dir = img_dir.replace('image', 'mask')
            mask_path = os.path.join(mask_dir, file)

            header = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(header)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)) if os.path.isfile(mask_path) else None
            spacing = header.GetSpacing()
            z_spacing = spacing[2]
            
            # 计算提取的通道
            betweenGapSlice = int(5 / z_spacing)
            depth, height, width = image.shape
            extract_ch = [ch for ch in range(0, depth, betweenGapSlice)]

            # 如果有掩码，添加有掩码的通道
            if os.path.isfile(mask_path) and mask is not None:
                for ch in range(0, depth):
                    if np.sum(mask[ch,:,:]) > 0:
                        extract_ch.append(ch)
            else:
                mask = None
            
            # 遍历提取的通道
            for ch in tqdm(extract_ch, desc=f"processing channels"):
                # 创建2.5D图像（当前通道及其相邻通道）
                tiff_image = np.zeros((height, width, depth_channel), dtype=np.float32)
                # if ch == 0, then extract the first 3 channels
                if ch == 0:
                    re_ch = [0, 1, 2]
                    for z, idx in enumerate(re_ch):
                        tiff_image[:, :, z] = image[idx, :, :]
                elif ch == depth-1:
                    re_ch = [depth-3, depth-2, depth-1]
                    for z, idx in enumerate(re_ch):
                        tiff_image[:, :, z] = image[idx, :, :]
                else:
                    for z, idx in enumerate(range(ch-1, ch+2)):
                        idx = np.clip(idx, 0, depth-1)
                        tiff_image[:, :, z] = image[idx, :, :]

                # HU窗口化（Hounsfield Units）
                np.clip(tiff_image, -1000, 400, out=tiff_image)
                # 归一化到0-255
                tiff_image = ((tiff_image + 1000) / 1400 * 255).astype('uint8')

                # 使用滑动窗口提取patch
                for y in range(0, height - patch_size + 1, stride_hw):
                    for x in range(0, width - patch_size + 1, stride_hw):
                        # 处理掩码
                        img_name = f"{file.replace('.nii.gz','')}_slice{ch}_y{y}_x{x}.png"

                        if mask is not None:
                            patch_mask = mask[ch, y:y+patch_size, x:x+patch_size]
                            # 如果patch中包含目标（掩码值大于0），则保存
                            if np.sum(patch_mask) > 0:
                                patch_mask[patch_mask > 0] = 255
                                cv2.imwrite(os.path.join(mask_save_dir, img_name), np.expand_dims(patch_mask, -1))
                                
                        # 保存patch图像
                        patch_img = tiff_image[y:y+patch_size, x:x+patch_size, :]
                        # np.save(os.path.join(img_save_dir, img_name.replace('.png', '.npy')), patch_img)
                        cv2.imwrite(os.path.join(img_save_dir, img_name), patch_img)
            del image
            del mask
            del header
            del tiff_image
            # 显式触发垃圾回收（可选）
            gc.collect()
            os.sync()
            time.sleep(1)
                
                        
                        
    # 处理完成 
    print('completion')
    
if __name__ == "__main__":
    # 解析命令行参数
    # 输入为 nifti_image_paths.csv，输出为切片后的 patch 数据集
    parser = argparse.ArgumentParser(description="data csv path")
    parser.add_argument(
        "-csv_path",
        default="/data/nifti_image_paths.csv",
        help="CSV 文件路径，需包含 path 列（或仅一列路径）",
    )
    parser.add_argument("-split", default=False, help="是否需要将数据分成训练、验证")
    parser.add_argument("-save_path", default=r"/data/yolo_dataset_249", help="save path of the dataset root")
    args = parser.parse_args()

    make_patch_data(args)