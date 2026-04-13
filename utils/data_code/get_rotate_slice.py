"""
旋转切面与CT序列相交模块

功能：给定一条物理坐标系下的空间水平线段作为旋转轴，
      创建绕该轴旋转的多个切面（0°到90°），
      将每个切面与CT序列相交的内容保存下来

用法示例：
    python rotated_plane_intersection.py --dicom /path/to/dicom \
        --axis_start 10,20,-100 --axis_end 100,20,-100 \
        --num_slices 5 --angle_range 90
"""

import numpy as np
import cv2
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
import json
import os
from pathlib import Path
from typing import Tuple, List, Optional
import argparse


def physical_to_voxel_coords(
    physical_coords: Tuple[float, float, float],
    image: sitk.Image
) -> Tuple[float, float, float]:
    """
    将物理坐标系转换为体素（像素）坐标系

    Args:
        physical_coords: 物理坐标 (x, y, z)，单位为mm
        image: SimpleITK图像对象

    Returns:
        voxel_coords: 体素坐标 (z, y, x)，与ct_data数组索引对应
    """
    continuous_index = image.TransformPhysicalPointToContinuousIndex(physical_coords)
    voxel_coords = (continuous_index[2], continuous_index[1], continuous_index[0])
    return voxel_coords


def voxel_to_physical_coords(
    voxel_coords: Tuple[float, float, float],
    image: sitk.Image
) -> Tuple[float, float, float]:
    """
    将体素坐标系转换为物理坐标系

    Args:
        voxel_coords: 体素坐标 (z, y, x)
        image: SimpleITK图像对象

    Returns:
        physical_coords: 物理坐标 (x, y, z)，单位为mm
    """
    # 体素坐标 (z, y, x) -> 连续索引 (x, y, z) -> 物理坐标
    continuous_index = (voxel_coords[2], voxel_coords[1], voxel_coords[0])
    physical_coords = image.TransformContinuousIndexToPhysicalPoint(continuous_index)
    return physical_coords


def load_dicom_series(dicom_dir: str) -> Tuple[np.ndarray, sitk.Image]:
    """
    加载DICOM序列

    Args:
        dicom_dir: DICOM文件夹路径

    Returns:
        ct_data: 3D CT体积数据 (z, y, x)
        image: SimpleITK图像对象
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(file_names)
    image = reader.Execute()
    ct_data = sitk.GetArrayFromImage(image).astype(np.float32)

    return ct_data, image


def load_nifti(nifti_path: str) -> Tuple[np.ndarray, sitk.Image]:
    """
    加载NIfTI文件

    Args:
        nifti_path: NIfTI文件路径

    Returns:
        ct_data: 3D CT体积数据 (z, y, x)
        image: SimpleITK图像对象（用于坐标转换）
    """
    import nibabel as nib
    nii_img = nib.load(nifti_path)
    ct_data = nii_img.get_fdata()

    if ct_data.ndim == 4:
        ct_data = ct_data.squeeze(-1)

    if ct_data.shape[2] <= ct_data.shape[0] and ct_data.shape[2] <= ct_data.shape[1]:
        ct_data = np.transpose(ct_data, (2, 1, 0))
    else:
        ct_data = np.transpose(ct_data, (2, 1, 0))

    ct_data = ct_data.astype(np.float32)

    image = sitk.GetImageFromArray(ct_data)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))

    return ct_data, image


def create_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    创建绕任意轴旋转的旋转矩阵（Rodrigues公式）

    Args:
        axis: 旋转轴单位向量 (3,)
        angle: 旋转角度（弧度）

    Returns:
        rotation_matrix: 3x3旋转矩阵
    """
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Rodrigues公式
    rotation_matrix = np.array([
        [cos_a + axis[0]**2 * (1 - cos_a),
         axis[0] * axis[1] * (1 - cos_a) - axis[2] * sin_a,
         axis[0] * axis[2] * (1 - cos_a) + axis[1] * sin_a],
        [axis[1] * axis[0] * (1 - cos_a) + axis[2] * sin_a,
         cos_a + axis[1]**2 * (1 - cos_a),
         axis[1] * axis[2] * (1 - cos_a) - axis[0] * sin_a],
        [axis[2] * axis[0] * (1 - cos_a) - axis[1] * sin_a,
         axis[2] * axis[1] * (1 - cos_a) + axis[0] * sin_a,
         cos_a + axis[2]**2 * (1 - cos_a)]
    ])

    return rotation_matrix


def create_rotated_slices(
    ct_data: np.ndarray,
    image: sitk.Image,
    axis_start_physical: Tuple[float, float, float],
    axis_end_physical: Tuple[float, float, float],
    angles: List[float],
    airway_diameter_mm: float = 30.0,
    window_center: float = -600,
    window_width: float = 1500,
    slice_resolution: Optional[Tuple[int, int]] = None,
    slice_size_mm: Tuple[float, float] = (300.0, 300.0),
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    创建绕给定轴旋转的多个切面

    Args:
        ct_data: 3D CT体积数据 (z, y, x)
        image: SimpleITK图像对象
        axis_start_physical: 旋转轴起点物理坐标 (x, y, z)，单位mm
        axis_end_physical: 旋转轴终点物理坐标 (x, y, z)，单位mm
        angles: 旋转角度列表（度），0度为水平面，90度为垂直面
        window_center: CT窗位 (HU)
        window_width: CT窗宽 (HU)
        slice_resolution: 输出切片分辨率 (width, height)，如指定则忽略slice_size_mm
        slice_size_mm: 切片物理尺寸 (width_mm, height_mm)，默认300x300mm确保覆盖CT

    Returns:
        slices: 切片图像列表 (BGR格式)
        metadata_list: 每个切片的元数据列表
    """
    # 计算旋转轴信息
    axis_start_voxel = np.array(physical_to_voxel_coords(axis_start_physical, image))
    axis_end_voxel = np.array(physical_to_voxel_coords(axis_end_physical, image))

    # 旋转轴中点
    axis_mid_voxel = (axis_start_voxel + axis_end_voxel) / 2

    # 旋转轴方向向量（物理空间）
    axis_direction_physical = np.array([
        axis_end_physical[0] - axis_start_physical[0],
        axis_end_physical[1] - axis_start_physical[1],
        axis_end_physical[2] - axis_start_physical[2]
    ])
    axis_direction_physical = axis_direction_physical / np.linalg.norm(axis_direction_physical)

    # 获取CT的spacing信息
    spacing = image.GetSpacing()  # (x, y, z)
    spacing = np.array([spacing[0], spacing[1], spacing[2]])

    # CT物理尺寸
    ct_physical_size = np.array([
        ct_data.shape[2] * spacing[0],  # X方向
        ct_data.shape[1] * spacing[1],  # Y方向
        ct_data.shape[0] * spacing[2],  # Z方向
    ])

    # CT包围盒对角线长度（确保任何方向旋转都能完整覆盖CT）
    ct_diagonal = np.linalg.norm(ct_physical_size)

    # 确定切片分辨率
    if slice_resolution is not None:
        slice_width_px, slice_height_px = slice_resolution
        slice_width_mm = slice_width_px * spacing[0]
        slice_height_mm = slice_height_px * spacing[1]
    else:
        # 使用CT对角线长度作为切片尺寸，确保无论旋转角度如何都能完整覆盖CT体积
        slice_width_mm = max(ct_diagonal, slice_size_mm[0])
        slice_height_mm = max(ct_diagonal, slice_size_mm[1])

        # 计算像素分辨率
        slice_width_px = int(round(slice_width_mm / spacing[0]))
        slice_height_px = int(round(slice_height_mm / spacing[2]))

    # 构建垂直于线段方向的基向量
    # 选择参考轴（不能与axis_direction平行）
    if abs(axis_direction_physical[2]) < 0.9:
        ref_axis = np.array([0.0, 0.0, 1.0])  # Z方向
    else:
        ref_axis = np.array([1.0, 0.0, 0.0])  # X方向

    # base_u 垂直于 axis_direction
    base_u = np.cross(ref_axis, axis_direction_physical)
    base_u = base_u / np.linalg.norm(base_u)
    # base_v 垂直于 axis_direction 和 base_u
    base_v = np.cross(axis_direction_physical, base_u)
    base_v = base_v / np.linalg.norm(base_v)

    # YZ面的法线方向（即X轴方向）
    yz_normal = np.array([1.0, 0.0, 0.0])

    slices = []
    metadata_list = []

    for angle in angles:
        # u_axis: 沿线段方向（固定，线段躺在平面上）
        u_axis = axis_direction_physical.copy()

        # w_axis: 垂直于u_axis且平行于YZ面的方向（切片的上下方向）
        # np.cross(u_axis, yz_normal) 自动得到垂直于u_axis且X分量为0的向量
        w_axis = np.cross(u_axis, yz_normal)
        w_norm = np.linalg.norm(w_axis)
        if w_norm < 1e-6:
            # 如果u_axis平行于X轴，使用备用方向
            w_axis = np.cross(u_axis, np.array([0.0, 1.0, 0.0]))
        w_axis = w_axis / np.linalg.norm(w_axis)

        # v_axis: 同时垂直于u_axis和w_axis（切片的左右方向）
        v_axis = np.cross(u_axis, w_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)

        # 创建切片网格（物理空间，以轴中点为中心）
        # 将物理尺寸转换为沿u_axis和v_axis方向的距离
        half_width = slice_width_mm / 2
        half_height = slice_height_mm / 2

        u_coords = np.linspace(-half_width, half_width, slice_width_px)
        v_coords = np.linspace(-half_height, half_height, slice_height_px)
        u, v = np.meshgrid(u_coords, v_coords)

        # 计算切片上每个点在物理空间中的坐标
        # 物理空间中，点 = 轴中点(物理) + u * u_axis + v * v_axis
        axis_mid_physical = np.array(voxel_to_physical_coords(tuple(axis_mid_voxel), image))

        points_physical = (
            axis_mid_physical[:, np.newaxis, np.newaxis] +
            u_axis[:, np.newaxis, np.newaxis] * u[np.newaxis, :, :] +
            v_axis[:, np.newaxis, np.newaxis] * v[np.newaxis, :, :]
        )

        # 将物理坐标转换为体素坐标用于插值
        # 注意：image.TransformPhysicalPointToContinuousIndex 接受的输入是 (x, y, z)
        # 返回的索引顺序是 (index_x, index_y, index_z)，需要转换为 (z, y, x)
        sampling_coords = np.zeros((3, slice_height_px, slice_width_px), dtype=np.float64)

        for i in range(slice_height_px):
            for j in range(slice_width_px):
                physical_point = (points_physical[0, i, j], points_physical[1, i, j], points_physical[2, i, j])
                continuous_index = image.TransformPhysicalPointToContinuousIndex(physical_point)
                # 转换顺序：continuous_index = (x, y, z) -> voxel = (z, y, x)
                sampling_coords[0, i, j] = continuous_index[2]
                sampling_coords[1, i, j] = continuous_index[1]
                sampling_coords[2, i, j] = continuous_index[0]

        # 三线性插值采样CT
        coords = np.array([
            sampling_coords[0].flatten(),
            sampling_coords[1].flatten(),
            sampling_coords[2].flatten()
        ])

        slice_values = map_coordinates(
            ct_data, coords, order=1, mode='constant', cval=0
        )
        slice_image = slice_values.reshape(slice_height_px, slice_width_px).astype(np.float32)

        # CT窗归一化
        hu_min = window_center - window_width / 2
        hu_max = window_center + window_width / 2
        slice_clipped = np.clip(slice_image, hu_min, hu_max)
        slice_norm = ((slice_clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)

        # 转换为彩色图像
        slice_bgr = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2BGR)

        # 在切片上绘制轴线位置
        # 轴线起点和终点在物理空间中
        # 由于线段完全躺在切片平面上，投影就是线段本身
        axis_half_length_mm = np.linalg.norm(
            np.array(axis_end_physical) - np.array(axis_start_physical)
        ) / 2

        axis_start_physical_pt = axis_mid_physical - axis_direction_physical * axis_half_length_mm
        axis_end_physical_pt = axis_mid_physical + axis_direction_physical * axis_half_length_mm

        # 计算线段端点在切片平面坐标系中的像素坐标
        # 由于线段沿线段方向u_axis，v方向投影为0
        def line_point_to_slice_coords(point, origin, u_ax, v_ax):
            """将物理空间中的点转换到切片像素坐标"""
            vec = point - origin
            u_physical = np.dot(vec, u_ax)
            v_physical = np.dot(vec, v_ax)
            u_pixel = (u_physical + half_width) / (2 * half_width) * slice_width_px
            v_pixel = (v_physical + half_height) / (2 * half_height) * slice_height_px
            return int(round(u_pixel)), int(round(v_pixel))

        # 计算线段端点在切片平面坐标系中的像素坐标（用于后续绘制轴线）
        start_pixel = line_point_to_slice_coords(axis_start_physical_pt, axis_mid_physical, u_axis, v_axis)
        end_pixel = line_point_to_slice_coords(axis_end_physical_pt, axis_mid_physical, u_axis, v_axis)

        # 气道直径参数（从函数参数传入）
        # 计算检测框角点（在切片平面坐标系中）
        # 以旋转轴为中心，根据气道直径在v_axis方向扩展
        half_axis_length_mm = axis_half_length_mm
        half_diameter_mm = airway_diameter_mm / 2

        # 检测框四个角点（物理空间，以轴中点为原点）
        # corner_u: u_axis方向的坐标（沿着轴）
        # corner_v: v_axis方向的坐标（垂直于轴，根据直径确定）
        bbox_corners_physical = [
            (-half_axis_length_mm, -half_diameter_mm),  # 左上
            ( half_axis_length_mm, -half_diameter_mm),  # 右上
            ( half_axis_length_mm,  half_diameter_mm),  # 右下
            (-half_axis_length_mm,  half_diameter_mm),  # 左下
        ]

        # 将检测框角点转换到像素坐标
        bbox_corners_pixel = []
        for corner_u, corner_v in bbox_corners_physical:
            # 角点的物理坐标
            corner_point = axis_mid_physical + corner_u * u_axis + corner_v * v_axis
            # 转换到切片像素坐标
            pixel_coords = line_point_to_slice_coords(corner_point, axis_mid_physical, u_axis, v_axis)
            bbox_corners_pixel.append(pixel_coords)

        # 绘制检测框（红色矩形）
        bbox_points = np.array(bbox_corners_pixel, dtype=np.int32)
        cv2.polylines(slice_bgr, [bbox_points], isClosed=True, color=(0, 0, 255), thickness=2)

        slices.append(slice_bgr)

        # 保存元数据
        metadata = {
            'angle': float(angle),
            'rotation_axis_start_physical': axis_start_physical,
            'rotation_axis_end_physical': axis_end_physical,
            'rotation_axis_mid_physical': tuple(axis_mid_physical),
            'rotation_axis_direction_physical': tuple(axis_direction_physical),
            'u_axis': tuple(u_axis.tolist()),
            'v_axis': tuple(v_axis.tolist()),
            'window_center': window_center,
            'window_width': window_width,
            'slice_size_mm': (float(slice_width_mm), float(slice_height_mm)),
            'slice_resolution': (slice_width_px, slice_height_px),
            'ct_shape': ct_data.shape,
            'ct_physical_size': tuple(ct_physical_size),
            'ct_diagonal_mm': float(ct_diagonal),
            'axis_line_start_pixel': start_pixel,
            'axis_line_end_pixel': end_pixel,
            'airway_diameter_mm': airway_diameter_mm,
            'bbox_corners_pixel': bbox_corners_pixel,
        }
        metadata_list.append(metadata)

    return slices, metadata_list


def save_rotated_slices(
    ct_path: str,
    axis_start_physical: Tuple[float, float, float],
    axis_end_physical: Tuple[float, float, float],
    output_dir: str,
    airway_diameter_mm: float = 30.0,
    num_slices: int = 5,
    angle_range: float = 90.0,
    window_center: float = -600,
    window_width: float = 1500,
    slice_resolution: Optional[Tuple[int, int]] = None,
    save_metadata: bool = True,
    save_individual: bool = True,
) -> Tuple[List[str], List[dict]]:
    """
    保存绕轴旋转的多个切面

    Args:
        ct_path: CT文件路径（DICOM文件夹或NIfTI文件）
        axis_start_physical: 旋转轴起点物理坐标 (x, y, z)
        axis_end_physical: 旋转轴终点物理坐标 (x, y, z)
        output_dir: 输出目录
        num_slices: 切面数量
        angle_range: 旋转角度范围（度），从0到该值
        window_center: CT窗位
        window_width: CT窗宽
        slice_resolution: 切片分辨率 (width, height)
        save_metadata: 是否保存元数据JSON
        save_individual: 是否保存每个切片的单独图像

    Returns:
        output_paths: 输出图像路径列表
        all_metadata: 所有切片的元数据列表
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载CT数据
    ct_path_obj = Path(ct_path)
    if ct_path_obj.is_dir():
        ct_data, image = load_dicom_series(ct_path)
    elif ct_path_obj.suffix in ('.nii', '.gz'):
        ct_data, image = load_nifti(ct_path)
    else:
        raise ValueError(f"Unsupported CT format: {ct_path}")

    # 计算角度列表
    angles = np.linspace(0, angle_range, num_slices).tolist()

    # 创建旋转切面
    slices, metadata_list = create_rotated_slices(
        ct_data, image,
        axis_start_physical, axis_end_physical,
        angles,
        airway_diameter_mm=airway_diameter_mm,
        window_center=window_center,
        window_width=window_width,
        slice_resolution=slice_resolution,
    )

    output_paths = []

    if save_individual:
        for i, (slice_bgr, metadata) in enumerate(zip(slices, metadata_list)):
            output_path = os.path.join(output_dir, f"rotated_slice_{i:03d}_angle_{metadata['angle']:.1f}.png")
            cv2.imwrite(output_path, slice_bgr)
            output_paths.append(output_path)

    # 保存所有切片的组合图像
    if slices:
        # 横向拼接所有切片
        combined = np.hstack(slices)
        combined_path = os.path.join(output_dir, "rotated_slices_combined.png")
        cv2.imwrite(combined_path, combined)
        output_paths.append(combined_path)

    # 保存元数据
    if save_metadata:
        metadata_path = os.path.join(output_dir, "rotated_slices_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)

    return output_paths, metadata_list


def parse_coords(coord_str: str) -> Tuple[float, float, float]:
    """解析坐标字符串 'x,y,z' 为元组"""
    parts = coord_str.split(',')
    if len(parts) != 3:
        raise ValueError(f"Invalid coordinate format: {coord_str}, expected 'x,y,z'")
    return tuple(map(float, parts))


def main():
    parser = argparse.ArgumentParser(
        description='旋转切面与CT序列相交 - 提取绕给定轴旋转的切面与CT的交面'
    )
    parser.add_argument('--input', '-i', default=r'/data/Mucus_origin_data/01-阳性数据/E0001046_20100112/DICOM/4F3DE55A/A357FDE0',
                        help='CT文件路径（DICOM文件夹或NIfTI文件）')
    parser.add_argument('--axis_start', '-s', default='-123.175,8.20523,-1259.77',
                        help='旋转轴起点物理坐标，格式: x,y,z (单位mm)')
    parser.add_argument('--axis_end', '-e', default='-69.4911,-13.0422,-1213.49', 
                        help='旋转轴终点物理坐标，格式: x,y,z (单位mm)')
    parser.add_argument('--diameter', '-d', type=float, default=6.7,
                        help='气道管直径，单位mm')
    parser.add_argument('--output', '-o', default='/workspace/rotate_results',
                        help='输出目录')
    parser.add_argument('--num_slices', '-n', type=int, default=36,
                        help='切面数量 (默认: 5)')
    parser.add_argument('--angle_range', '-a', type=float, default=360.0,
                        help='旋转角度范围，度 (默认: 90)')
    parser.add_argument('--window_center', '-wc', type=float, default=-600,
                        help='CT窗位 (默认: -600)')
    parser.add_argument('--window_width', '-ww', type=float, default=1500,
                        help='CT窗宽 (默认: 1500)')
    parser.add_argument('--resolution', type=str, default=None,
                        help='切片分辨率，格式: width,height (默认: 使用CT原始大小)')
    parser.add_argument('--no_metadata', action='store_true',
                        help='不保存元数据JSON')
    parser.add_argument('--no_individual', action='store_true',
                        help='不保存单独的切片图像')

    args = parser.parse_args()

    # 解析坐标
    axis_start = parse_coords(args.axis_start)
    axis_end = parse_coords(args.axis_end)

    # 解析分辨率
    resolution = None
    if args.resolution:
        parts = args.resolution.split(',')
        resolution = (int(parts[0]), int(parts[1]))

    # 执行处理
    print(f"输入CT: {args.input}")
    print(f"旋转轴起点 (物理坐标): {axis_start}")
    print(f"旋转轴终点 (物理坐标): {axis_end}")
    print(f"气道直径: {args.diameter} mm")
    print(f"旋转角度范围: 0° - {args.angle_range}°")
    print(f"切面数量: {args.num_slices}")
    print(f"输出目录: {args.output}")
    print(f"CT窗: WC={args.window_center}, WW={args.window_width}")

    output_paths, metadata_list = save_rotated_slices(
        ct_path=args.input,
        axis_start_physical=axis_start,
        axis_end_physical=axis_end,
        output_dir=args.output,
        airway_diameter_mm=args.diameter,
        num_slices=args.num_slices,
        angle_range=args.angle_range,
        window_center=args.window_center,
        window_width=args.window_width,
        slice_resolution=resolution,
        save_metadata=not args.no_metadata,
        save_individual=not args.no_individual,
    )

    print(f"\n处理完成!")
    print(f"生成 {len(output_paths)} 个文件:")
    for path in output_paths:
        print(f"  - {path}")

    if not args.no_metadata:
        print(f"元数据已保存: {os.path.join(args.output, 'rotated_slices_metadata.json')}")


if __name__ == '__main__':
    main()