"""
Create patch datasets from medical image data.

Reads a CSV file containing paths to NIfTI (.nii.gz) data files.
Loads NIfTI images and corresponding masks, applies HU windowing,
extracts 2.5D patches, and splits the dataset into train and validation sets.
Patches are saved as .png images (3-channel 2.5D), and masks as .png (if present).
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

    # Read paths from CSV, split into train/valid (75% / 25%)
    df = pd.read_csv(args.csv_path)
    # Support column named "path", or use the first column
    path_col = "path" if "path" in df.columns else df.columns[0]
    paths = df[path_col].dropna().astype(str).tolist()
    # Create output directory
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

        # Configuration parameters
        patch_size = 128  # Patch size
        stride_hw = 64    # Sliding window stride
        depth_channel = 3  # 2.5D depth channels

        # new_df = defaultdict(list)

        # Process each dataset (train, valid); dataset contains full image paths
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

                # Mask path: same directory, replace "image" with "mask", same filename
                img_dir = os.path.dirname(image_path)
                mask_dir = img_dir.replace('image', 'mask')
                mask_path = os.path.join(mask_dir, file)

                header = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(header)
                mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)) if os.path.isfile(mask_path) else None
                spacing = header.GetSpacing()
                z_spacing = spacing[2]

                # Compute extraction channels
                betweenGapSlice = int(5 / z_spacing)
                depth, height, width = image.shape
                extract_ch = [ch for ch in range(0, depth, betweenGapSlice)]

                # If mask exists, add channels that contain annotations
                if os.path.isfile(mask_path) and mask is not None:
                    for ch in range(0, depth):
                        try:
                            if np.sum(mask[ch,:,:]) > 0:
                                extract_ch.append(ch)
                        except Exception as e:
                            print(f"Caught error: {e}")
                            continue
                else:
                    mask = None

                # Iterate over extraction channels
                for ch in tqdm(extract_ch, desc=f"processing channels"):
                    # Create 2.5D image (current channel + adjacent channels)
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

                    # HU windowing (Hounsfield Units)
                    np.clip(tiff_image, -1000, 400, out=tiff_image)
                    # Normalize to 0-255
                    tiff_image = ((tiff_image + 1000) / 1400 * 255).astype('uint8')

                    # Extract patches with sliding window
                    for y in range(0, height - patch_size + 1, stride_hw):
                        for x in range(0, width - patch_size + 1, stride_hw):
                            img_name = f"{file.replace('.nii.gz','')}_slice{ch}_y{y}_x{x}.png"

                            if mask is not None:
                                patch_mask = mask[ch, y:y+patch_size, x:x+patch_size]
                                # Save mask patch only if it contains a target (value > 0)
                                if np.sum(patch_mask) > 0:
                                    patch_mask[patch_mask > 0] = 255
                                    cv2.imwrite(os.path.join(mask_save_dir, img_name), np.expand_dims(patch_mask, -1))

                            # Save patch image
                            patch_img = tiff_image[y:y+patch_size, x:x+patch_size, :]
                            # np.save(os.path.join(img_save_dir, img_name.replace('.png', '.npy')), patch_img)
                            cv2.imwrite(os.path.join(img_save_dir, img_name), patch_img)
                del image
                del mask
                del header
                del tiff_image
                gc.collect()
                os.sync()
                time.sleep(0.5)

    if not args.split:
        n = len(paths)
        print(f"total: {n}")
        patch_size = 128  # Patch size
        stride_hw = 64    # Sliding window stride
        depth_channel = 3  # 2.5D depth channels

        # new_df = defaultdict(list)

        img_save_dir = os.path.join(savePath, 'All', 'images')
        mask_save_dir = os.path.join(savePath, 'All', 'masks')
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)

        for f, image_path in enumerate(paths):

            file = os.path.basename(image_path)
            print('-'*20 + f"processing All--{file.replace('.nii.gz','')}--{f+1}/{n}" + '-'*20)
            # new_df['filename'].append(file)
            # new_df['set_type'].append(saveSetName[i])

            # Mask path: same directory, replace "image" with "mask", same filename
            img_dir = os.path.dirname(image_path)
            mask_dir = img_dir.replace('image', 'mask')
            mask_path = os.path.join(mask_dir, file)

            header = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(header)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)) if os.path.isfile(mask_path) else None
            spacing = header.GetSpacing()
            z_spacing = spacing[2]

            # Compute extraction channels
            betweenGapSlice = int(5 / z_spacing)
            depth, height, width = image.shape
            extract_ch = [ch for ch in range(0, depth, betweenGapSlice)]

            # If mask exists, add channels that contain annotations
            if os.path.isfile(mask_path) and mask is not None:
                for ch in range(0, depth):
                    if np.sum(mask[ch,:,:]) > 0:
                        extract_ch.append(ch)
            else:
                mask = None

            # Iterate over extraction channels
            for ch in tqdm(extract_ch, desc=f"processing channels"):
                # Create 2.5D image (current channel + adjacent channels)
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

                # HU windowing (Hounsfield Units)
                np.clip(tiff_image, -1000, 400, out=tiff_image)
                # Normalize to 0-255
                tiff_image = ((tiff_image + 1000) / 1400 * 255).astype('uint8')

                # Extract patches with sliding window
                for y in range(0, height - patch_size + 1, stride_hw):
                    for x in range(0, width - patch_size + 1, stride_hw):
                        img_name = f"{file.replace('.nii.gz','')}_slice{ch}_y{y}_x{x}.png"

                        if mask is not None:
                            patch_mask = mask[ch, y:y+patch_size, x:x+patch_size]
                            # Save mask patch only if it contains a target (value > 0)
                            if np.sum(patch_mask) > 0:
                                patch_mask[patch_mask > 0] = 255
                                cv2.imwrite(os.path.join(mask_save_dir, img_name), np.expand_dims(patch_mask, -1))

                        # Save patch image
                        patch_img = tiff_image[y:y+patch_size, x:x+patch_size, :]
                        # np.save(os.path.join(img_save_dir, img_name.replace('.png', '.npy')), patch_img)
                        cv2.imwrite(os.path.join(img_save_dir, img_name), patch_img)
            del image
            del mask
            del header
            del tiff_image
            gc.collect()
            os.sync()
            time.sleep(1)

    print('completion')

if __name__ == "__main__":
    # Parse command-line arguments
    # Input: nifti_image_paths.csv, output: sliced patch dataset
    parser = argparse.ArgumentParser(description="data csv path")
    parser.add_argument(
        "-csv_path",
        default="/data/nifti_image_paths.csv",
        help="CSV file path, must contain a 'path' column (or a single path column)",
    )
    parser.add_argument("-split", default=False, help="Whether to split data into train/valid")
    parser.add_argument("-save_path", default=r"/data/yolo_dataset_249", help="save path of the dataset root")
    args = parser.parse_args()

    make_patch_data(args)
