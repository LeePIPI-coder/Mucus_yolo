# import torch
# import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc
# from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from PIL import Image
from tqdm import tqdm
import cv2
import nibabel as nib
import json
import msvcrt  # Windows 专用
from gzip import compress, decompress
from zipfile import ZipFile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
# import lz4.frame
from typing import Sequence
from utils.simAPSMaskUpload.StorageReader.SeriesList import Position, SeriesItem
from utils.simAPSMaskUpload.ConsoleApp.UploadMask import __generate_init_mask_json, __generate_lesion_file_and_append_lesion, __generate_annot_json_if_not_existed
from openpyxl import Workbook, load_workbook
from pathlib import Path
from utils.SQL.SqlcipherStorageReader import SqlcipherStorageReader
from datetime import datetime
import zipfile
# import lz4.frame
from utils.logging import get_logger
from ultralytics import YOLO
import pydicom

def extra_ct(root_dir):
    ct_lists = []
    if len(os.listdir(root_dir)) > 2:
        for i in os.listdir(root_dir):
            if i.endswith('dcm'):
                path = os.path.join(root_dir, i)
                # _, ct_info = process_ct(path)
                _, ct_info = process_ct_pro(path)
                if ct_info == None:
                    # ct_lists = None
                    break
                ct_lists.append(ct_info)
    else:
        for i in os.listdir(root_dir):
            if i.endswith('dcm'):
                path = os.path.join(root_dir, i)
                _, ct_info = process_ct_pro(path)
                ct_lists.append(ct_info)
    return ct_lists

def process_ct_pro(dicom_path, mode='normalized'):

    try:
        ct = pydicom.dcmread(dicom_path, force=True)  # force=True 避免 header 不一致时报错
        image = ct.pixel_array.astype(np.float32)
    except Exception as e:
        print(f"[错误] 读取像素失败: {dicom_path},\n{e}")
        return None, None

    # HU 转换
    slope = getattr(ct, 'RescaleSlope', 1)
    intercept = getattr(ct, 'RescaleIntercept', 0)
    image = image * slope + intercept

    # 元信息
    dicom_info ={
    "meta": {
        "PatientID": getattr(ct, "PatientID", None),
        "PatientName": str(getattr(ct, "PatientName", "")),
        "StudyID": getattr(ct, "StudyID", None),
        "AccessionNumber": getattr(ct, "AccessionNumber", None),
        "StudyInstanceUID": getattr(ct, "StudyInstanceUID", None),
        "SeriesNumber": getattr(ct, "SeriesNumber", None),
        "InstanceNumber": getattr(ct, "InstanceNumber", None),
        "SeriesInstanceUID": getattr(ct, "SeriesInstanceUID", None),
        "ImagePositionPatient": getattr(ct, "ImagePositionPatient", None),
        "ImageOrientationPatient": getattr(ct, "ImageOrientationPatient", None),
        "PixelSpacing": getattr(ct, "PixelSpacing", None),
        "SliceThickness": getattr(ct, "SliceThickness", None),
        "SpacingBetweenSlices": getattr(ct, "SpacingBetweenSlices", None),
        "SOPInstanceUID": getattr(ct, "SOPInstanceUID", None),
    },
    "image": image
    }

    return image, dicom_info

def get_ct_info(ct: list[str]):
    '''
        根据提取的ct列表来获取ct中的各个指标参数
    '''
    patient_id = ct[0]['meta']['PatientID']
    patient_name = ct[0]['meta']['PatientName']
    study_id = ct[0]['meta']['StudyID']
    access_number = ct[0]['meta']['AccessionNumber']
    study_instance_uid = ct[0]['meta']['StudyInstanceUID']
    series_number = ct[0]['meta']['SeriesNumber']
    series_instance_uid = ct[0]['meta']['SeriesInstanceUID']
    sopInstanceUid = ct[0]['meta']['SOPInstanceUID']
    return patient_id, patient_name, study_id, access_number, study_instance_uid, series_number, series_instance_uid, sopInstanceUid

def normalize_to_uint8(img):
    # img: 2D numpy float or int
    mn = np.nanmin(img)
    mx = np.nanmax(img)
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - mn) / (mx - mn)
    arr = (scaled * 255.0).astype(np.uint8)
    return arr

def load_dicom_series(dicoms):
    def key_fn(item):
        ds = item['meta']
        return ds.get('InstanceNumber', None) or ds.get('SliceLocation', None) or 0
    dicoms.sort(key=key_fn)
    slices = [item['image'] for item in dicoms]
    vol = np.stack(slices, axis=0)  # shape (Z, H, W)
    return vol

def detect_volume(vol, out_images_dir: Path, model=None, patch_size=128, stride=None):
    if stride is None:
        stride = patch_size
    # batch_size_dict = {128: 16, 64: 64}
    # batch_size = batch_size_dict.get(stride)
    # vol shape: (Z, H, W)
    Z, H, W = vol.shape
    records = []

    batch_size = (H // stride)**2
    # 新增：保存原始整张 slice 的目录
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # 批处理相关
    batch_patches = []
    all_predictions = []

    # 进度条初始化
    pb = tqdm(total=Z, desc='Detecting', unit='slice') if tqdm is not None else None
    
    for z in range(Z):

        slice_img = vol[z]
        
        img_u8 = normalize_to_uint8(slice_img)
        # ensure 3 channels for YOLO (BGR)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

        # 保存整张 slice 图作为背景（一次）
        slice_fname = f"slice_z{z:04d}.png"

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                x2 = x + patch_size
                y2 = y + patch_size
                patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                src_x2 = min(x2, W)
                src_y2 = min(y2, H)
                src = img_bgr[y:src_y2, x:src_x2]
                h_src, w_src = src.shape[:2]
                patch[0:h_src, 0:w_src] = src
                fname = f"patch_z{z:04d}_x{x:05d}_y{y:05d}.png"
                # fpath = out_images_dir / fname
                # if z == 319:
                #     cv2.imwrite(str(fpath), patch)
                records.append({
                    'filename': fname,
                    'slice': int(z),
                    'x': int(x),
                    'y': int(y),
                    'patch_w': patch_size,
                    'patch_h': patch_size,
                    'orig_w': int(W),
                    'orig_h': int(H),
                })

                # 累积patch到批次
                batch_patches.append(patch)

                # 当批次达到batch_size时进行检测
                if len(batch_patches) >= batch_size:
                    # 进行模型推理
                    results = model(batch_patches, verbose=False)
                    all_predictions.extend(results)
                    # 清空批次
                    batch_patches = []

        pb.update(1)
        
    # 处理剩余的patch
    if len(batch_patches) > 0:
        batch_images = np.array(batch_patches)
        results = model(batch_images)
        all_predictions.extend(results)

    return records, all_predictions


def get_mask_Pos(mask):
    rows = np.any(mask, axis=1)  # 哪些行有非零值
    cols = np.any(mask, axis=0)  # 哪些列有非零值

    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]  # 上下边界
        x_min, x_max = np.where(cols)[0][[0, -1]]  # 左右边界

        top_left = (int(x_min), int(y_min), 0)
        bottom_right = (int(x_max), int(y_max), 1)
    else:
        # 掩码为空
        top_left = (0, 0, 1)
        bottom_right = (0, 0, 1)
    return top_left, bottom_right

def save_lesions_json(json_lesions: list[dict], output_file: str):
    """
    保存病变列表到 JSON 文件
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_lesions, f, ensure_ascii=False, indent=4)

import pydicom
import numpy as np

def get_precise_affine(ct_lists: Sequence[dict]):
    # 1. 提取并强制转换为 NumPy float 数组
    # 使用 np.array(..., dtype=np.float64) 确保后续矩阵运算不会因为类型冲突报错
    origin = np.array(ct_lists[0]['meta'].get('ImagePositionPatient'), dtype=np.float64)
    orient = np.array(ct_lists[0]['meta'].get('ImageOrientationPatient'), dtype=np.float64)
    spacing = np.array(ct_lists[0]['meta'].get('PixelSpacing'), dtype=np.float64) # [row_spacing, col_spacing]
    
    # 2. 计算真实的层间向量 (Z轴)
    pos1 = np.array(ct_lists[0]['meta'].get('ImagePositionPatient'), dtype=np.float64)
    pos2 = np.array(ct_lists[1]['meta'].get('ImagePositionPatient'), dtype=np.float64)
    z_axis_vector = pos2 - pos1 
    
    # 3. 计算方向向量
    # 因为 orient 已经是 numpy 数组，切片出来的也是 numpy 数组
    x_axis = orient[:3]
    y_axis = orient[3:]
    
    # 4. 构建仿射矩阵
    affine = np.eye(4, dtype=np.float64)
    
    # 执行元素级乘法：numpy array * float
    affine[:3, 0] = x_axis * spacing[1] # DICOM 列间隔对应图像的 X 方向
    affine[:3, 1] = y_axis * spacing[0] # DICOM 行间隔对应图像的 Y 方向
    affine[:3, 2] = z_axis_vector       # 层间偏移向量包含了 Z 方向的旋转和间距
    affine[:3, 3] = origin
    
    return affine

import nibabel as nib
import numpy as np

def convert_bbox_to_patient_pos(affine, detect_boxes):
    detect_bbox = {}
    for i, detect_box in enumerate(detect_boxes):
        
        bbox_min = [detect_box[0], detect_box[1], detect_box[4]]
        bbox_max = [detect_box[2], detect_box[3], detect_box[4] + 3]
        pixel_coords = [bbox_max, bbox_min]
        
        physical_pts = []
        for coord in pixel_coords:
            # 将坐标转换为 4x1 向量 [x, y, z, 1]
            v = np.array([coord[0], coord[1], coord[2], 1.0])
            # 矩阵乘法
            p = affine @ v
            physical_pts.append(p[:3])
        
        physical_pts = np.array(physical_pts)

        roi_patientPos_min = np.min(physical_pts, axis=0)
        roi_patientPos_max = np.max(physical_pts, axis=0)
        roi_patientPos_min = f"({roi_patientPos_min[0]:.4f}, {roi_patientPos_min[1]:.4f}, {roi_patientPos_min[2]:.4f})"
        roi_patientPos_max = f"({roi_patientPos_max[0]:.4f}, {roi_patientPos_max[1]:.4f}, {roi_patientPos_max[2]:.4f})"
        detect_bbox[f'{i}'] = [roi_patientPos_min, roi_patientPos_max]
    return detect_bbox

def Seg_Upload(input_path: str, start_time: str, model_path: str, annotation: str, weather_jumpy: bool):
    
    logger = get_logger(input_path)
    sql = SqlcipherStorageReader(input_path)
    logger.info("正在提取要检测的文件目录。。。")
    all_patient_list = sql.get_all_series_list()
    
    all_patient = [patient for patient in all_patient_list if datetime.strptime(patient.updated_at, "%Y-%m-%d %H:%M:%S") >= datetime.strptime(start_time, "%Y-%m-%d")]
    
    # all_patient = [patient for patient in all_patient_list if patient.patient_id == '09641883' and patient.series_number == 303]
    
    # 加载对用的权重文件
    model = YOLO(str(model_path))

    # patient_dirs = get_dicom_paths(input_path) 
    patient_num = len(all_patient)
    
    # 对所有的dicom文件目录进行遍历
    for i, patient in enumerate(all_patient):
        if i < 57:
            continue
        annotation_exist_flog = 0
        ct_lists = []
        patient_dir = patient.abs_path
        results_dir = patient_dir + '\\stor\\results'
        logger.info('-' * 10 + f'开始处理第{i+1}个病例/共{patient_num}个病例' + '-' * 10)
        logger.info(f"正在处理的目录：{patient_dir}")
        # 在对应的dicom文件目录下创建 stor/results目录
        results_dir = os.path.join(patient_dir, "stor", "results")
        os.makedirs(results_dir, exist_ok=True)
        json_path = results_dir + '\\' + 'lesionAnnot3D.json'
        
        
        # 查看是否标签是否已经写入,若写入则跳过改目录
        if os.path.exists(json_path) and weather_jumpy:
            with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data['study'][0]['series'][0].get('lesion') is not None:
                            for i, data_ann in enumerate(data['study'][0]['series'][0]['lesion']):
                                if data_ann['userAnnotComment']['annotation'] == annotation:
                                    logger.debug(f"目录:{patient_dir}已经存在对应的掩码文件,跳过该目录")
                                    annotation_exist_flog = 1
        if annotation_exist_flog == 1:
            continue

        # 根据dicom文件目录来提取dicom并转换为numpy数组
        ct_lists = extra_ct(patient_dir)
        if len(ct_lists) == 1:
            continue
        ct_lists.sort(key=lambda x: x['meta']['ImagePositionPatient'][2])
        vol = load_dicom_series(ct_lists)# shape (Z, H, W)
        
        records, all_predict = detect_volume(vol, out_images_dir=Path('E:\Hangzhou_workspace\LJR_workspace\Mucus_yolo_Slice'), model=model, patch_size=128, stride=None)
        detect_boxes = []
        for record, predict in zip(records, all_predict):
            z = record['slice']
            x_S = record['x']
            y_S = record['y']
            # 判断是否存在检测目标
            if len(predict.boxes) != 0:
                boxes = predict.boxes.xyxy.cpu().numpy()
                scores = predict.boxes.conf.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    ox1 = int(x1) + x_S
                    oy1 = int(y1) + y_S
                    ox2 = int(x2) + x_S
                    oy2 = int(y2) + y_S
                    detect_boxes.append([ox1, oy1, ox2, oy2, z])
        
        # 根据ct_lists字典来提取特定参数用于后续生成json文件及其他指标计算
        (patient_id, patient_name, study_id, access_number, study_instance_uid,
                series_number, series_instance_uid, sopInstanceUid) = get_ct_info(ct_lists)
        
        # 计算仿射变换矩阵
        affine = get_precise_affine(ct_lists)
        
        # 转换检测框为患者坐标系下的最小/最大角点
        all_detect_bbox = convert_bbox_to_patient_pos(affine, detect_boxes)

        # # 如果病灶类型为3D, 则计算患者坐标系下的最小/最大角点
        # patient_pos_min, patient_pos_max = compute_patient_bbox_from_ct_lists(ct_lists)  


        # 设置变量series_info
        series_info = SeriesItem(
            patient_id=patient_id,
            patient_name=patient_name,
            study_id=study_id,
            access_number=access_number,
            study_instance_uid=study_instance_uid,
            series_number=series_number,
            series_instance_uid=series_instance_uid,
            abs_path=patient_dir,
            # patient_pos_max=patient_pos_max,#
            # patient_pos_min=patient_pos_min,#
            sopInstanceUid = sopInstanceUid,
            # top_left=top_left,
            # bottom_right=bottom_right,
            rows=patient.rows,
            cols=patient.cols,
            sop_instance_count=patient.sop_instance_count,
            # patient_vol = patient_vol,
            updated_at = None
        )

        # 如果json文件不存在，根据series_info来生成对应的json内容
        json, start_lesion_index  = __generate_annot_json_if_not_existed(results_dir, json_path, series_info)
        
        # 将压缩后的掩码数据 gz_compress_mask 再压缩成.psegmaskz文件，并在json文件中添加对应的掩码信息
        __generate_lesion_file_and_append_lesion(
            dim_X=512,
            dim_Y=512,
            dim_Z=505,
            gz_compressed_mask=vol, 
            series_info=series_info,
            results_dir=results_dir,
            mask_name="mucus",   
            json_lesions=json['study'][0]['series'][0]['lesion'],
            start_lesion_index=start_lesion_index,
            detect_bbox=all_detect_bbox
            )
        save_lesions_json(json, f'{json_path}')
    # logger.info(f"csv文件保存在{input_path}/patients_results.xlsx")
    logger.info(f"训练日志存储在{input_path}/train.log")
    logger.info("---------已处理完成！------------")


if __name__ == '__main__':
    '''
    功能：
        根据上传文件根目录中的数据库文件来分析并生成目录下各个序列对应的掩码
        并将掩码保存在与各个序列同目录的 stor/results 目录下
        保存的内容有对应的annotarion-json文件以及掩码的.psegmaskz
    参数：
        root_dir: 要处理的文件目录
        start_time: 开始处理的时间
        data_type : 要处理的病灶类型
        Annotation: 保存成的病灶标签类型
        weather_jumpy: 是否跳过已经存在要写入Annotation的病例
    '''    
    # ------------用argparse参数输入----------------------
    parser = argparse.ArgumentParser(description="预测病灶掩膜")
    parser.add_argument('--root_dir', type=str, default=r'F:\kernel_adaption\kernel_adaption_NYS',
                        help='要处理的文件目录')
    parser.add_argument('--start_time', type=str, default='2022-1-1',
                        help='文件上传的时间')
    parser.add_argument('--model_path', type=str, default=r'weights\best_241.pt',
                        help='要检测的病灶类型')
    parser.add_argument('--Annotation', type=str, default='mucus-20260309',
                        help='写入json的annotation内容')
    parser.add_argument('--weather_jumpy', type=bool, default=False,
                    help='如果已经存在要写入的annotation,是否跳过')
    opt = parser.parse_args()

    Seg_Upload(opt.root_dir, opt.start_time, opt.model_path, opt.Annotation, opt.weather_jumpy) 

    #-------------用终端提示输入-----------------------------
    # while True:
    #     root_dir = input("请输入要处理的文件目录:")
    #     start_time = input("请输入要处理序列的时间节点:")
    #     data_type = input("请输入要检测的病灶类型:")
    #     Annotation = input("模型检测的掩码标签:")

    #     Seg_Upload(root_dir, start_time, data_type, Annotation)
    #     print("分割完成！")