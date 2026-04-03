import nibabel as nib
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
from utils.simAPSMaskUpload.StorageReader.SeriesList import Position, SeriesItem
from utils.simAPSMaskUpload.ConsoleApp.UploadMask import __generate_init_mask_json, __generate_annot_json_if_not_existed,__generate_lesion_file_and_append_lesion_mask
from utils.SQL.SqlcipherStorageReader import SqlcipherStorageReader
from gzip import compress
from utils.logging import get_logger
import json
import SimpleITK as sitk
import pydicom

logger = get_logger(r"E:\Hangzhou_workspace\LJR_workspace\Mucus_yolo\logs\split_niigz_label")


def save_lesions_json(json_lesions: list[dict], output_file: str):
    """
    保存病变列表到 JSON 文件
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_lesions, f, ensure_ascii=False, indent=4)

def compute_bounding_box_and_physical_coords(single_region_data, affine_matrix):
    """
    计算3D分割掩码的边界框并将其转换为物理坐标
    
    Args:
        single_region_data: 3D分割掩码数据
        affine_matrix: 仿射变换矩阵
    
    Returns:
        tuple: (bbox_min, bbox_max, roi_patientPos_min, roi_patientPos_max) 其中
               bbox_min/max 是体素坐标系中的边界框坐标
               roi_patientPos_min/max 是物理坐标系中的边界框坐标
    """
    # 计算边界框：找到非零元素的索引
    xs, ys, zs = np.where(single_region_data)
    if len(zs) > 0 and len(ys) > 0 and len(xs) > 0:
        # 计算体素坐标系中的边界框
        bbox_min = [xs.min(), ys.min(), zs.min()]  # [x_min, y_min, z_min]
        bbox_max = [xs.max()+1, ys.max()+1, zs.max()+1]  # [x_max, y_max, z_max]
        maskbbox = [bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1], bbox_max[2] - bbox_min[2]]
        for i, num in enumerate(maskbbox):
            if num < 4:
                lang = 4 - num
                lang_remain = lang % 2
                lang_num = lang // 2
                if lang_num > 0:
                    bbox_min[i] -= lang_num
                    bbox_max[i] += lang_num
                if lang_remain == 1:
                    bbox_min[i] -= 1
                    
        # 使用仿射矩阵将体素坐标转换为物理坐标
        pixel_coords = [bbox_min, bbox_max]
        bbox_min = f'{str(bbox_min[0])},{str(bbox_min[1])},{str(bbox_min[2])}'
        bbox_max = f'{str(bbox_max[0])},{str(bbox_max[1])},{str(bbox_max[2])}'
        affine_matrix[2][3] = -affine_matrix[2][3]
        
        physical_pts = []
        for coord in pixel_coords:
            # 将坐标转换为 4x1 向量 [x, y, z, 1]
            v = np.array([-coord[0], -coord[1], coord[2], -1.0])
            # 矩阵乘法
            p = affine_matrix @ v
            physical_pts.append(p[:3])  # 只保留物理坐标 [x, y, z]
        
        physical_pts = np.array(physical_pts)
        
        roi_patientPos_min = np.min(physical_pts, axis=0)
        roi_patientPos_max = np.max(physical_pts, axis=0)
        
        roi_patientPos_min = [roi_patientPos_min[0], roi_patientPos_min[1], roi_patientPos_min[2]]
        roi_patientPos_max = [roi_patientPos_max[0], roi_patientPos_max[1], roi_patientPos_max[2]]
        
        return bbox_min, bbox_max, roi_patientPos_min, roi_patientPos_max

def split_niigz_labels(input_file_path, dicom_dir, sop_instance_count):
    """
    将包含多个粘液栓标签的Niigz文件分离成多个单独的文件
    每个单独的粘液栓标签值都设为1
    
Args:
        input_file_path: 输入的niigz文件路径
    """
    results_dir =os.path.join(dicom_dir, "stor", "results")
    json_path = os.path.join(results_dir, 'lesionAnnot3D.json')
    
    file = next(f for f in os.listdir(dicom_dir) if f.endswith(".dcm"))
    ds = pydicom.dcmread(os.path.join(dicom_dir, file))
    patient_id = ds.PatientID
    patient_name = ds.PatientName
    study_id = ds.StudyID
    access_number = ds.AccessionNumber
    study_instance_uid = ds.StudyInstanceUID
    series_number = ds.SeriesNumber
    series_instance_uid = ds.SeriesInstanceUID
    abs_path = dicom_dir
    sop_instance_uid = ds.SOPInstanceUID
    row = ds.Rows
    cols = ds.Columns
    sop_instance_count = sop_instance_count
    update=None
    
    # 读取原始niigz文件
    img = nib.load(str(input_file_path))
    data = img.get_fdata()
    
    data = np.where(data > 0, 1, 0).astype(np.uint8)
    labeled_array, num_features = ndimage.label(data)
    
    logger.info(f"在 {input_file_path} 中发现 {num_features} 个连通的粘液栓区域")

    # 遍历每个连通域（粘液栓）
    for i in range(1, num_features + 1):
        
        affine = img.affine.copy()
        lesion_dict = {}
        userAnnotComment_dict = {}
        # 创建只包含当前连通域的数据
        single_region_data = np.where(labeled_array == i, 255, 0).astype(np.uint8)
        
        # 计算边界框和物理坐标
        bbox_min, bbox_max, roi_patientPos_min, roi_patientPos_max = compute_bounding_box_and_physical_coords(single_region_data, affine)
        patient_pos_max = Position(roi_patientPos_max[0], roi_patientPos_max[1], roi_patientPos_max[2])
        patient_pos_min = Position(roi_patientPos_min[0], roi_patientPos_min[1], roi_patientPos_min[2])
        # lesion_file = results_dir + f"\\lesionAnnot3D-{i:03d}.{data.shape[0]}x{data.shape[1]}x{data.shape[2]}.psegmaskz"

        series_info = SeriesItem(
        patient_id=patient_id,
        patient_name=patient_name,
        study_id=study_id,
        access_number=access_number,
        study_instance_uid=study_instance_uid,
        series_number=series_number,
        series_instance_uid=series_instance_uid,
        abs_path=dicom_dir,
        sopInstanceUid = sop_instance_uid,
        rows=row,
        cols=cols,
        patient_pos_max=patient_pos_max,
        patient_pos_min=patient_pos_min,
        top_left=bbox_min,
        bottom_right=bbox_max,
        sop_instance_count=sop_instance_count,
        updated_at = '',
        study_date_time = ''
        ) 

        json_content, start_lesion_index  = __generate_annot_json_if_not_existed(results_dir, json_path, series_info)
        # single_region_data = single_region_data[:,:,::-1]
        binary_mask_bytes = encode_mask_Bit2Byte(single_region_data.transpose(2, 1, 0))
        gz_compressed_mask = compress(binary_mask_bytes)
        
        __generate_lesion_file_and_append_lesion_mask(
            dim_X=single_region_data.shape[0],
            dim_Y=single_region_data.shape[1],
            dim_Z=single_region_data.shape[2],
            gz_compressed_mask=gz_compressed_mask, 
            series_info=series_info,
            results_dir=results_dir,
            mask_name=f"Mask{i}",   
            json_lesions=json_content['study'][0]['series'][0]['lesion'],
            start_lesion_index=start_lesion_index,
        )
        save_lesions_json(json_content, f'{json_path}')
    logger.info(f"总共创建了 {num_features} 个分离的标签文件在 {results_dir}")

def encode_mask_Bit2Byte(mask_array: np.ndarray) -> bytes:
    """
    将三维掩膜 [z, y, x] 编码为字节流
    mask_array: numpy.ndarray, 取值 {0,1}，形状 [z, y, x]
    return: 压缩后的字节流
    """
    assert mask_array.ndim == 3, "mask_array 必须是三维的 [z, y, x]"
    
    z, y, x = mask_array.shape
    # 判断每行需要多少字节（位存储，1bit一个像素）
    row_bytes_len = (x + 7) // 8   # 向上取整

    mask_bytes = bytearray()

    # 按照 slice -> row -> col 顺序打包
    for zi in range(z):
        for yi in range(y):  
            # row_bits = mask_array[zi, :, yi]
            row_bits = mask_array[zi, yi, :]   # 取出一行 (x,)

            byte_val = 0
            bit_count = 0
            for bit in row_bits:
                # 每一位写入当前字节（小端位序，最低位先写）
                if bit:
                    byte_val |= (1 << bit_count)
                bit_count += 1

                # 一个字节凑满就存入
                if bit_count == 8:
                    mask_bytes.append(byte_val)
                    byte_val = 0
                    bit_count = 0

            # 如果最后不足8位，补零
            if bit_count > 0:
                mask_bytes.append(byte_val)

    # 转成bytes并压缩
    mask_bytes = bytes(mask_bytes)
    return mask_bytes

def find_depth_dir(dicom_path):
    if "阳性数据" in dicom_path or "广医粘液栓标注" in dicom_path:
        for item in os.listdir(dicom_path):
            if not os.path.isdir(os.path.join(dicom_path, item)):
                continue
            for sub_item in os.listdir(os.path.join(dicom_path, item)):
                if os.path.isdir(os.path.join(dicom_path, item, sub_item)):
                    for sub_sub_item in os.listdir(os.path.join(dicom_path, item, sub_item)):
                        if os.path.isdir(os.path.join(dicom_path, item, sub_item, sub_sub_item)):
                            return os.path.join(dicom_path, item, sub_item, sub_sub_item)
    else:
        for item in os.listdir(dicom_path):
            if not os.path.isdir(os.path.join(dicom_path, item)):
                continue
            for sub_item in os.listdir(os.path.join(dicom_path, item)):
                if os.path.isdir(os.path.join(dicom_path, item, sub_item)):
                    return os.path.join(dicom_path, item, sub_item)

def split_label(sql_path, origin_data_root):

    patient_root = Path(origin_data_root)
    patient_list = list(patient_root.iterdir())
    patient_dir_dict = {}
    Dir = []
    for patient in patient_list:
        if patient.name.startswith('E000'):
            patient_dir_dict[patient.name] = patient
        else:
            Dir.append(patient)
    for patient_dir in Dir:
        for patient in patient_dir.iterdir():
            patient_dir_dict[patient.name] = patient
            
    # 读取数据库文件
    sql = SqlcipherStorageReader(sql_path)
    all_patient_list = sql.get_all_series_list()
    
    j = 0
    for i, patient in enumerate(all_patient_list):

        patient_ID = patient.patient_id + "_" + patient.study_date_time.split(' ')[0].replace('-', '')
        logger.info(f"正在处理患者 {i+1}/{len(all_patient_list)}: {patient.patient_id}")
        logger.info(f"患者路径: {patient.abs_path}")
        
        sop_instance_count = patient.sop_instance_count
        patient_dir = Path(patient.abs_path)
        results_dir = patient_dir / 'stor/results'
        # 使用 glob 匹配所有以 .nii.gz 结尾的文件
        try:
            patient_Path = patient_dir_dict[patient_ID]
        except:
            logger.warning(f"患者 {patient_ID} 掩码不存在")
            continue
        
        mask_niigz = list(patient_Path.glob('*.nii.gz'))[0]
        
        #----------------测试用，用完删----------------------
        # mask_niigz = r"S:\AVIEWDB\mucus_plug\dcm\2009\11\ST_40351_00000151\SE_00002_00000151\stor\results\lesionAnnot3D-000.nii.gz"
        
        split_niigz_labels(mask_niigz, patient_dir, sop_instance_count)
    
    
    
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="将包含多个粘液栓标签的Niigz文件分离成多个单独的文件")
    parser.add_argument("--origin_data_dir", "-i", type=str, help="输入的niigz文件路径或包含niigz文件的目录", default="S:\Algo_space\LJR\Mucus_origin_data")
    parser.add_argument("--sql_path", "-s", type=str, help="数据库路径", default="S:\AVIEWDB\mucus_plug")
    
    args = parser.parse_args()
    
    split_label(args.sql_path, args.origin_data_dir)
