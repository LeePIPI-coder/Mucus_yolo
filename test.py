import cv2
import os
import numpy as np
import argparse
import tqdm
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from utils.utils import load_dicom_series, pre_processing, post_processing, coord_vox2pat

def predict(image_path, inference_model):
    patch_size=128
    stride_hw=64
    resize_to=512
    depth_channel=3
    threshold=0.01

    global new_df
    new_df = defaultdict(list)

    image, info = load_dicom_series(image_path)
    image = pre_processing(image)
    depth, height, width = image.shape

    for ch in tqdm.tqdm(range(depth), desc="Inference Processing"):
        patch_2p5d = np.zeros((height, width, depth_channel), dtype=np.float32)
        for z, re_ch in enumerate(range(ch-1, ch+2)):
            re_ch = np.clip(re_ch, 0, depth-1)
            patch_2p5d[:, :, z] = image[re_ch, :, :]

        for y in range(0, height - patch_size + 1, stride_hw):
            for x in range(0, width - patch_size + 1, stride_hw):
                patch_img = patch_2p5d[y:y+patch_size, x:x+patch_size, :]
                patch_img_resized = cv2.resize(patch_img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)

                outputs = inference_model.predict(source=patch_img_resized, save=False, imgsz=512, verbose=False)
                if len(outputs[0].boxes) == 0:
                    continue

                boxes = outputs[0].boxes.xyxy.detach().cpu().numpy()
                scores = outputs[0].boxes.conf.detach().cpu().numpy()

                for box, score in zip(boxes, scores):
                    if score < threshold:
                        continue

                    x1_orig = int(box[0] * patch_size / resize_to) + x
                    y1_orig = int(box[1] * patch_size / resize_to) + y
                    x2_orig = int(box[2] * patch_size / resize_to) + x
                    y2_orig = int(box[3] * patch_size / resize_to) + y

                    diax = abs(x2_orig - x1_orig)
                    diay = abs(y2_orig - y1_orig)
                    diaz = max(abs(x2_orig - x1_orig), abs(y2_orig - y1_orig))

                    z1_orig = max(ch - (diaz // 2), 0)
                    z2_orig = min(ch + (diaz // 2), depth - 1)
                    
                    center_x = int(x1_orig + x2_orig) / 2
                    center_y = int(y1_orig + y2_orig) / 2
                    center_z = int(ch)

                    world_coord_center = np.array(coord_vox2pat([center_x, center_y, center_z], info['origin'], info['spacing'], info['direction']))
                    world_coord_min = np.array(coord_vox2pat([x1_orig, y1_orig, z1_orig], info['origin'], info['spacing'], info['direction']))
                    world_coord_max = np.array(coord_vox2pat([x2_orig, y2_orig, z2_orig], info['origin'], info['spacing'], info['direction']))
                    
                    new_df['StudyInstanceUID'].append(info['StudyInstanceUID'])
                    new_df['SeriesInstanceUID'].append(info['SeriesInstanceUID'])
                    new_df['roi_patientPos_min_x'].append(world_coord_min[0])
                    new_df['roi_patientPos_min_y'].append(world_coord_min[1])
                    new_df['roi_patientPos_min_z'].append(world_coord_min[2])
                    new_df['roi_patientPos_max_x'].append(world_coord_max[0])
                    new_df['roi_patientPos_max_y'].append(world_coord_max[1])
                    new_df['roi_patientPos_max_z'].append(world_coord_max[2])
                    new_df['roi_patientPos_center_x'].append(world_coord_center[0])
                    new_df['roi_patientPos_center_y'].append(world_coord_center[1])
                    new_df['roi_patientPos_center_z'].append(world_coord_center[2])
                    new_df['roi_patient_diameter_x'].append(abs(world_coord_max[0] - world_coord_min[0]))
                    new_df['roi_patient_diameter_y'].append(abs(world_coord_max[1] - world_coord_min[1]))
                    new_df['roi_patient_diameter_z'].append(abs(world_coord_max[2] - world_coord_min[2]))
                    new_df['LesionType'].append('ELesionAnnotType_ROI_3D')
                    new_df['detector_score'].append(score)
                    new_df['x'].append(world_coord_center[0]) # pixel coordinate
                    new_df['y'].append(world_coord_center[1]) # pixel coordinate
                    new_df['z'].append(world_coord_center[2]) # pixel coordinate
                    new_df['diameter'].append(max(max(diax, diay), diaz)) # pixel level
                    new_df['diameter_x'].append(diax) # pixel level
                    new_df['diameter_y'].append(diay) # pixel level
                    new_df['diameter_z'].append(diaz) # pixel level
                    new_df['path'].append(image_path)
                    # new_df['x1_orig'].append(x1_orig) # pixel coordinate
                    # new_df['y1_orig'].append(y1_orig) # pixel coordinate
                    # new_df['x2_orig'].append(x2_orig) # pixel coordinate
                    # new_df['y2_orig'].append(y2_orig) # pixel coordinate

    new_df = pd.DataFrame(new_df)
    filtered_df = post_processing(new_df)
    return filtered_df

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model_pth", 
                      default='/home/LJR/Mucus_project/demo_mucusAlgorithms/weights/det_mucus.pt', 
                      help="The fold model weight")
    args.add_argument("-i", "--image_path", 
                      default="/data/Mucus_origin_data/广医粘液栓标注/E0001501_20111227/DICOM/73916312/8F131430",
                      help="dicom series's image folder path", type=str)
    args = args.parse_args()
    
    root_df = defaultdict(list)
    inference_model = YOLO(args.model_pth)
    for root, dirs, files in os.walk(args.image_path):
        if len(dirs) == 0:
            dcm_files = [file for file in files if file.endswith('.dcm')]
            if len(dcm_files) == 0 or len(dcm_files) < 10:
                continue
            print(f"[INFO] DICOM series path: {root}")
            filtered_df = predict(root, inference_model)
            for key, value in filtered_df.items():
                root_df[key].extend(value)
    root_df = pd.DataFrame(root_df)
    root_df.to_csv('./results.csv', index=False)