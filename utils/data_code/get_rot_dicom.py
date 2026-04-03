import numpy as np
import cv2
from scipy.ndimage import map_coordinates
import nibabel as nib
from typing import Tuple, List
import SimpleITK as sitk
import json


def create_perpendicular_slice_from_line(
    ct_data: np.ndarray,  # 3D CT volume [z, y, x]
    line_start: Tuple[float, float, float],  # [x1, y1, z1] 起始点
    line_end: Tuple[float, float, float],    # [x2, y2, z2] 终止点
    bbox_3d: Tuple[float, float, float, float, float, float],  # [x1, y1, z1, x2, y2, z2] 3D标注框
    image_size: Tuple[int, int] = (512, 512)  # 输出切面图像大小
) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    """
    根据3D空间中的线段和标注框，生成垂直于该线段的切面图像
    
    Args:
        ct_data: 3D CT volume [z, y, x]
        line_start: 线段起始点 [x1, y1, z1]
        line_end: 线段终止点 [x2, y2, z2]
        bbox_3d: 3D标注框 [x1, y1, z1, x2, y2, z2]
        image_size: 输出切面图像大小 (width, height)
    
    Returns:
        tuple: (slice_image, bbox_2d, transformation_matrix)
               slice_image: 切面图像
               bbox_2d: 2D标注框 [x1, y1, x2, y2]
               transformation_matrix: 从3D到2D的变换矩阵
    """
    # 计算线段的方向向量
    direction = np.array(line_end) - np.array(line_start)
    direction = direction / np.linalg.norm(direction)
    
    # 计算线段中点作为切面中心
    center_point = np.array(line_start) + 0.5 * (np.array(line_end) - np.array(line_start))
    
    # 构建垂直于线段方向的平面坐标系
    # 找到两个垂直于方向向量的单位向量作为新坐标系的基向量
    # 选择一个不与方向向量平行的向量
    if abs(direction[2]) < 0.9:
        temp_axis = np.array([0, 0, 1])
    else:
        temp_axis = np.array([1, 0, 0])
    
    # 使用叉积计算第一个正交向量
    u_axis = np.cross(temp_axis, direction)
    u_axis = u_axis / np.linalg.norm(u_axis)
    
    # 使用叉积计算第二个正交向量
    v_axis = np.cross(direction, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)
    
    # 定义输出图像网格
    width, height = image_size
    u_range = np.linspace(-width/2, width/2, width)
    v_range = np.linspace(-height/2, height/2, height)
    U, V = np.meshgrid(u_range, v_range)
    
    # 将2D网格转换为3D坐标
    X = center_point[0] + U * u_axis[0] + V * v_axis[0]
    Y = center_point[1] + U * u_axis[1] + V * v_axis[1]
    Z = center_point[2] + U * u_axis[2] + V * v_axis[2]
    
    # 确保坐标在有效范围内
    X = np.clip(X, 0, ct_data.shape[2] - 1)
    Y = np.clip(Y, 0, ct_data.shape[1] - 1)
    Z = np.clip(Z, 0, ct_data.shape[0] - 1)
    
    # 使用三线性插值获取切面图像
    coords = np.array([Z.flatten(), Y.flatten(), X.flatten()])
    slice_values = map_coordinates(ct_data, coords, order=1, mode='constant', cval=0)
    slice_image = slice_values.reshape(height, width).astype(np.float32)
    
    # 将3D标注框投影到切面上
    bbox_3d = np.array(bbox_3d)
    bbox_min = bbox_3d[:3]  # [x1, y1, z1]
    bbox_max = bbox_3d[3:]  # [x2, y2, z2]
    
    # 计算8个顶点
    vertices = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],  # 左下前
        [bbox_max[0], bbox_min[1], bbox_min[2]],  # 右下前
        [bbox_min[0], bbox_max[1], bbox_min[2]],  # 左上前
        [bbox_max[0], bbox_max[1], bbox_min[2]],  # 右上前
        [bbox_min[0], bbox_min[1], bbox_max[2]],  # 左下后
        [bbox_max[0], bbox_min[1], bbox_max[2]],  # 右下后
        [bbox_min[0], bbox_max[1], bbox_max[2]],  # 左上后
        [bbox_max[0], bbox_max[1], bbox_max[2]]   # 右上后
    ])
    
    # 将3D顶点投影到2D切面上
    projected_vertices = []
    for vertex in vertices:
        # 计算相对于切面中心的向量
        rel_vertex = vertex - center_point
        # 投影到u轴和v轴上
        u_proj = np.dot(rel_vertex, u_axis)
        v_proj = np.dot(rel_vertex, v_axis)
        # 转换为图像坐标
        x_img = u_proj + width / 2
        y_img = v_proj + height / 2
        projected_vertices.append([x_img, y_img])
    
    projected_vertices = np.array(projected_vertices)
    
    # 计算包围所有投影顶点的2D边界框
    x_coords = projected_vertices[:, 0]
    y_coords = projected_vertices[:, 1]
    
    x_min = int(max(0, np.floor(np.min(x_coords))))
    y_min = int(max(0, np.floor(np.min(y_coords))))
    x_max = int(min(width - 1, np.ceil(np.max(x_coords))))
    y_max = int(min(height - 1, np.ceil(np.max(y_coords))))
    
    bbox_2d = (x_min, y_min, x_max, y_max)
    
    # 构建变换矩阵（从3D世界坐标到2D切面坐标）
    transform_matrix = np.array([
        [u_axis[0], u_axis[1], u_axis[2], width/2 - np.dot(center_point, u_axis)],
        [v_axis[0], v_axis[1], v_axis[2], height/2 - np.dot(center_point, v_axis)],
        [0, 0, 0, 1]
    ])
    
    return slice_image, bbox_2d, transform_matrix


def save_perpendicular_slice(
    ct_volume_path: str,
    line_start: Tuple[float, float, float],
    line_end: Tuple[float, float, float],
    bbox_3d: Tuple[float, float, float, float, float, float],
    output_image_path: str,
    output_bbox_path: str = None
):
    """
    保存垂直于指定线段的切面图像和对应的2D标注框
    
    Args:
        ct_volume_path: 3D CT体积文件路径 (如.nii格式)
        line_start: 线段起始点 [x1, y1, z1]
        line_end: 线段终止点 [x2, y2, z2]
        bbox_3d: 3D标注框 [x1, y1, z1, x2, y2, z2]
        output_image_path: 输出切面图像路径
        output_bbox_path: 输出2D标注框信息路径（可选）
    """
    # 加载CT体积数据
    nii_img = nib.load(ct_volume_path)
    ct_data = nii_img.get_fdata()
    
    # 确保数据是 [z, y, x] 格式
    if ct_data.ndim == 4 and ct_data.shape[-1] == 1:
        ct_data = ct_data.squeeze(-1)
    
    # 生成垂直切面
    slice_image, bbox_2d, transform_matrix = create_perpendicular_slice_from_line(
        ct_data, line_start, line_end, bbox_3d
    )
    
    # 归一化图像到0-255范围以便保存
    slice_image_norm = ((slice_image - slice_image.min()) / 
                       (slice_image.max() - slice_image.min()) * 255).astype(np.uint8)
    
    # 保存切面图像
    cv2.imwrite(output_image_path, slice_image_norm)
    
    # 如果需要保存2D标注框信息
    if output_bbox_path:
        import json
        bbox_info = {
            "bbox_2d": bbox_2d,
            "line_start": line_start,
            "line_end": line_end,
            "bbox_3d": bbox_3d,
            "transform_matrix": transform_matrix.tolist()
        }
        with open(output_bbox_path, 'w') as f:
            json.dump(bbox_info, f, indent=2)
    
    return slice_image, bbox_2d

def load_dicom_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    assert series_ids, "No DICOM found"

    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(file_names)

    image = reader.Execute()

    img = sitk.GetArrayFromImage(image)  # (D, H, W)
    spacing = image.GetSpacing()[::-1]   # (D, H, W)

    return img.astype(np.float32), image


def get_mask(mask_path: str):
    with open(mask_path, 'r') as f:
        bbox_info = json.load(f)
        if bbox_info['study'][0]['series'][0].get('lesion') is not None:
            for i, data_ann in enumerate(bbox_info['study'][0]['series'][0]['lesion']):
                if data_ann['lesionType'] == 'ELesionAnnotType_ROI_3D':
                    mask_pos_min = data_ann["roi_patientPos_min"][1:-1].split(',')
                    bbox_3d_min = list(map(float, mask_pos_min))
                    mask_pos_max = data_ann["roi_patientPos_max"][1:-1].split(',')
                    bbox_3d_max = list(map(float, mask_pos_max))
                    bbox_3d = (bbox_3d_min[0], bbox_3d_min[1], bbox_3d_min[2], bbox_3d_max[0], bbox_3d_max[1], bbox_3d_max[2])
                if data_ann['lesionType'] == 'ELesionAnnotType_Simple_3D_Line':
                    line_pos = data_ann['points']
                    line = [list(map(float, p.split(','))) for p in line_pos.split('|')]
                # TODO 后续加上处理其他标注框
            return bbox_3d, line
                    
            
            
# 使用示例
if __name__ == "__main__":
    # 示例用法
    
    # CT图像序列
    CT_img = r"E:\aveiwdata\dcm\2011\01\ST_2414_00000121\SE_00005_00000786"
    # 读取CT图像序列
    
    ct_data, image = load_dicom_series(CT_img)
        
    mask_path = r"E:\aveiwdata\dcm\2011\01\ST_2414_00000121\SE_00005_00000786\stor\results\lesionAnnot3D.json"
    # 读取标注框信息
    bbox_3d_physical, line = get_mask(mask_path)
    
    bbox_3d_min = image.TransformPhysicalPointToContinuousIndex(bbox_3d_physical[:3])
    bbox_3d_max = image.TransformPhysicalPointToContinuousIndex(bbox_3d_physical[3:])
    line_start = image.TransformPhysicalPointToContinuousIndex(line[0])
    line_end = image.TransformPhysicalPointToContinuousIndex(line[1])
    
    bbox_3d = (bbox_3d_min[0], bbox_3d_min[1], bbox_3d_min[2], bbox_3d_max[0], bbox_3d_max[1], bbox_3d_max[2])
    
    # 生成并保存垂直切面
    try:
        slice_image, bbox_2d = save_perpendicular_slice(
            ct_volume_path="path/to/your/ct_volume.nii",
            line_start=line_start,
            line_end=line_end,
            bbox_3d=bbox_3d,
            output_image_path="perpendicular_slice.png",
            output_bbox_path="bbox_2d_info.json"
        )
        
        print(f"切面图像已保存至: perpendicular_slice.png")
        print(f"2D标注框信息: {bbox_2d}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")