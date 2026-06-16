#!/usr/bin/env python3
"""
将左肺和右肺的nii.gz掩码合并为一个整体肺部掩码。
左肺保持原值，右肺值加1区分，或直接合并为二值掩码。
"""

import nibabel as nib
import numpy as np
import os
import json
import re

def get_json_point(DATA_DIR):
    """从 lesionAnnot3D.json 提取标注点的中心坐标。

    - maskPos "(x1,y1,z1)-(x2,y2,z2)" → 体素坐标，取中点
    - points "x1,y1,z1|x2,y2,z2" (Dir/De 线段) → 世界坐标 (mm)，取端点

    返回:
      mask_info: [{"name": str, "center_world": (x, y, z)|None, "maskPos_str": str}, ...]
      line_info: [{"name": str, "type": "Dir"|"De", "endpoints_world": ((x1,y1,z1), (x2,y2,z2))}, ...]
    """
    json_path = os.path.join(DATA_DIR, "lesionAnnot3D.json")
    mask_info = []
    line_info = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for lesion in data['study'][0]['series'][0]['lesion']:
        userAnnot = lesion.get('userAnnotComment', {})
        if not userAnnot:
            continue
        annotation = userAnnot.get('annotation', '')

        # --- Mask 标注：maskPos 是 AVIEW 体素坐标，与 nibabel 体素坐标系不同 ---
        # 世界坐标 (roi_patientPos) 是可靠的地面真值，统一用世界坐标转换
        maskPos_str = lesion.get('maskPos', '')
        if maskPos_str:
            m = re.findall(r"\((\d+),(\d+),(\d+)\)", maskPos_str)
            # 世界坐标用 roi_patientPos 提供的中点
            pos_min_str = lesion.get('roi_patientPos_min', '')
            pos_max_str = lesion.get('roi_patientPos_max', '')
            if pos_min_str and pos_max_str:
                p_min = np.array([float(x) for x in pos_min_str.strip('()').split(',')])
                p_max = np.array([float(x) for x in pos_max_str.strip('()').split(',')])
                center_world = tuple(((p_min + p_max) / 2).tolist())
            else:
                center_world = None
            mask_info.append({
                "name": annotation,
                "center_world": center_world,       # 世界坐标 (mm) — 用于转换
                "maskPos_str": maskPos_str,         # 原始 AVIEW 体素 (仅供参考)
            })
            continue

        # --- Dir / De 线段标注：points 是世界坐标 (mm) 浮点数 ---
        points_str = lesion.get('points', '')
        if points_str:
            pts = re.findall(r"(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)", points_str)
            if len(pts) == 2:
                p1 = tuple(float(x) for x in pts[0])  # (x, y, z) world
                p2 = tuple(float(x) for x in pts[1])
                seg_type = "Dir" if "Dir" in annotation else "De"
                line_info.append({
                    "name": annotation,
                    "type": seg_type,
                    "endpoints_world": (p1, p2),
                })

    return mask_info, line_info
               
def get_bbox(mask):
    """计算掩码中非零体素的外接矩形 (bounding box)。
    mask: 3D numpy array [x, y, z], 0=背景, >0=前景
    返回: (x_min, x_max, y_min, y_max, z_min, z_max) 均为 inclusive index
    """
    x_idx, y_idx, z_idx = np.where(mask > 0)
    return (x_idx.min(), x_idx.max(),
            y_idx.min(), y_idx.max(),
            z_idx.min(), z_idx.max())


def crop_by_bbox(nii, bbox, margin=0):
    """用 bounding box 裁剪 nii 图像。
    nii: nibabel Nifti1Image (data 为 [x, y, z] 顺序)
    bbox: (x_min, x_max, y_min, y_max, z_min, z_max)
    margin: 边界外扩的体素数，默认 0
    返回: (cropped_nii, actual_bbox)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    data = nii.get_fdata()
    shape = data.shape   # [x, y, z]

    cropped_data = data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]

    # 更新 affine：新原点是旧原点 + bbox 起点的物理偏移
    new_origin = nib.affines.apply_affine(nii.affine, [x_min, y_min, z_min])
    new_affine = nii.affine.copy()
    new_affine[:3, 3] = new_origin

    # actual_bbox = (x_min, x_max, y_min, y_max, z_min, z_max)s
    return nib.Nifti1Image(cropped_data, new_affine, nii.header), bbox


def original_voxel_to_cropped_voxel(pts, bbox):
    """将原始 DCM/NIfTI 体素坐标转换为裁剪后的体素坐标。

    pts: shape (N, 3) numpy array, 每行按 (x, y, z) 顺序
    bbox: (x_min, x_max, y_min, y_max, z_min, z_max)
    返回: shape (N, 3) numpy array, 裁剪后体素坐标 (x, y, z)
    """
    x_min, _x_max, y_min, _y_max, z_min, _z_max = bbox
    offset = np.array([x_min, y_min, z_min])
    return pts - offset


def world_to_cropped_voxel(pts_world, orig_affine, bbox):
    """将 AVIEW 世界坐标 (mm) 转换为裁剪后的体素坐标。

    使用与 split_niigz_label.py 一致的逆向变换：
      x_vox = (px + ox) / |sx|,  y_vox = (py + oy) / |sy|,  z_vox = (pz - oz) / sz

    pts_world: shape (N, 3) numpy array, 每行按 (x, y, z) 顺序 (AVIEW 世界坐标)
    orig_affine: 原始未裁剪图像的 affine 矩阵 (4x4)
    bbox: (x_min, x_max, y_min, y_max, z_min, z_max)
    返回: shape (N, 3) numpy array, 裁剪后体素坐标 (x, y, z)
    """
    sx = abs(orig_affine[0, 0])
    sy = abs(orig_affine[1, 1])
    sz = orig_affine[2, 2]
    ox = orig_affine[0, 3]
    oy = orig_affine[1, 3]
    oz = orig_affine[2, 3]

    x_min, _x_max, y_min, _y_max, z_min, _z_max = bbox

    # 逆向: AVIEW世界 (px, py, pz) → nibabel [x_vox, y_vox, z_vox]
    x_vox = (pts_world[:, 0] + ox) / sx
    y_vox = (pts_world[:, 1] + oy) / sy
    z_vox = (pts_world[:, 2] - oz) / sz

    # 裁剪: 转为 (x, y, z) 顺序输出
    return np.column_stack([
        x_vox - x_min,
        y_vox - y_min,
        z_vox - z_min,
    ])


def cropped_voxel_to_world(pts_cropped, cropped_affine):
    """将裁剪后的体素坐标转换回世界坐标 (mm)。

    pts_cropped: shape (N, 3) numpy array, 每行按 (x, y, z) 顺序
    cropped_affine: 裁剪后图像的 affine 矩阵 (4x4)
    返回: shape (N, 3) numpy array, 世界坐标 (x, y, z)
    """
    return nib.affines.apply_affine(cropped_affine, pts_cropped)


def get_lung_dcm(DATA_DIR):
    LEFT_LUNG = os.path.join(DATA_DIR, "LungSeg.Obj.LtLung.nii.gz")
    RIGHT_LUNG = os.path.join(DATA_DIR, "LungSeg.Obj.RtLung.nii.gz")
    DCM = os.path.join(DATA_DIR, "image.nii.gz")
    OUTPUT = os.path.join(DATA_DIR, "LungSeg.Obj.Lung.nii.gz")


    left_nii = nib.load(LEFT_LUNG)
    right_nii = nib.load(RIGHT_LUNG)
    dcm_nii = nib.load(DCM)

    left_data = left_nii.get_fdata().astype(np.int16)
    right_data = right_nii.get_fdata().astype(np.int16)

    # 合并：左肺=1，右肺=2，背景=0
    merged = np.zeros_like(left_data, dtype=np.int16)
    merged[left_data > 0] = 1
    merged[right_data > 0] = 2

    # 保存合并掩码
    merged_nii = nib.Nifti1Image(merged, left_nii.affine, left_nii.header)
    nib.save(merged_nii, OUTPUT)
    print(f"已保存合并掩码到: {OUTPUT}")
    print(f"  左肺体素 (label=1): {np.sum(merged == 1)}")
    print(f"  右肺体素 (label=2): {np.sum(merged == 2)}")

    # 计算肺部外接矩形
    bbox = get_bbox(merged)
    print(f"  肺部外接矩形 (x, y, z): [{bbox[0]}:{bbox[1]}, {bbox[2]}:{bbox[3]}, {bbox[4]}:{bbox[5]}]")

    cropped_nii, crop_bbox = crop_by_bbox(dcm_nii, bbox, margin=5)
    cropped_output = os.path.join(DATA_DIR, "lung_image.nii.gz")
    nib.save(cropped_nii, cropped_output)
    print(f"已保存裁剪后掩码到: {cropped_output}")
    print(f"  实际裁剪 bbox (x, y, z): [{crop_bbox[0]}:{crop_bbox[1]}, {crop_bbox[2]}:{crop_bbox[3]}, {crop_bbox[4]}:{crop_bbox[5]}]")

    return {
        "orig_affine": dcm_nii.affine,
        "cropped_affine": cropped_nii.affine,
        "lung_bbox": bbox,            # 不含 margin 的肺部 bbox
        "crop_bbox": crop_bbox,       # 含 margin 的实际裁剪 bbox
    }

if __name__ == "__main__":
    data_dir = "/data/Mucus_data/E0001425_20100927"

    # 1. 合并肺部掩码 + 裁剪 CT
    info = get_lung_dcm(data_dir)

    # 2. 提取 JSON 中的标注点
    mask_info, line_info = get_json_point(data_dir)

    # 3. 转换到裁剪后体素坐标
    crop_bbox = info["crop_bbox"]   # 用含 margin 的实际 bbox
    orig_affine = info["orig_affine"]

    print("\n========== Mask 中心点 ==========")
    for m in mask_info:
        cw = m["center_world"]
        if cw:
            cw_cropped = world_to_cropped_voxel(
                np.array([cw]), orig_affine, crop_bbox
            )[0]
            print(f"  {m['name']}: 世界中心=({cw[0]:.1f}, {cw[1]:.1f}, {cw[2]:.1f})  →  裁剪体素=({cw_cropped[0]:.1f}, {cw_cropped[1]:.1f}, {cw_cropped[2]:.1f})")
        else:
            print(f"  {m['name']}: 无世界坐标，maskPos={m['maskPos_str']} (仅供参考，不可直接用于nibabel体素)")

    print("\n========== 线段端点 (Dir/De) ==========")
    for l in line_info:
        ep1, ep2 = l["endpoints_world"]
        ep1_cropped = world_to_cropped_voxel(np.array([ep1]), orig_affine, crop_bbox)[0]
        ep2_cropped = world_to_cropped_voxel(np.array([ep2]), orig_affine, crop_bbox)[0]
        print(f"  {l['name']}({l['type']}):")
        print(f"    端点1 世界=({ep1[0]:.1f}, {ep1[1]:.1f}, {ep1[2]:.1f})  →  裁剪体素=({ep1_cropped[0]:.1f}, {ep1_cropped[1]:.1f}, {ep1_cropped[2]:.1f})")
        print(f"    端点2 世界=({ep2[0]:.1f}, {ep2[1]:.1f}, {ep2[2]:.1f})  →  裁剪体素=({ep2_cropped[0]:.1f}, {ep2_cropped[1]:.1f}, {ep2_cropped[2]:.1f})")

    # 4. 写入 txt 文件
    output_txt = os.path.join(data_dir, "annotation_points_cropped.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("# 裁剪后体素坐标 (x, y, z)，基于 lung_image.nii.gz\n")
        f.write(f"# crop_bbox (x, y, z): [{crop_bbox[0]}:{crop_bbox[1]}, {crop_bbox[2]}:{crop_bbox[3]}, {crop_bbox[4]}:{crop_bbox[5]}]\n\n")

        f.write("[Mask]\n")
        for m in mask_info:
            cw = m["center_world"]
            if cw:
                cv = world_to_cropped_voxel(np.array([cw]), orig_affine, crop_bbox)[0]
                f.write(f"{m['name']} {cv[0]:.1f} {cv[1]:.1f} {cv[2]:.1f}\n")
        f.write("\n")

        # Dir/De 按编号分组配对输出
        pairs = {}
        for l in line_info:
            num = l["name"][0]  # "1Dir" → "1"
            pairs.setdefault(num, {})[l["type"]] = l

        for num in sorted(pairs.keys()):
            dir_entry = pairs[num].get("Dir")
            de_entry = pairs[num].get("De")
            if dir_entry:
                ep1, ep2 = dir_entry["endpoints_world"]
                c1 = world_to_cropped_voxel(np.array([ep1]), orig_affine, crop_bbox)[0]
                c2 = world_to_cropped_voxel(np.array([ep2]), orig_affine, crop_bbox)[0]
                f.write(f"[{num}Dir] {c1[0]:.1f} {c1[1]:.1f} {c1[2]:.1f} | {c2[0]:.1f} {c2[1]:.1f} {c2[2]:.1f}\n")
            if de_entry:
                ep1, ep2 = de_entry["endpoints_world"]
                c1 = world_to_cropped_voxel(np.array([ep1]), orig_affine, crop_bbox)[0]
                c2 = world_to_cropped_voxel(np.array([ep2]), orig_affine, crop_bbox)[0]
                f.write(f"[{num}De] {c1[0]:.1f} {c1[1]:.1f} {c1[2]:.1f} | {c2[0]:.1f} {c2[1]:.1f} {c2[2]:.1f}\n")

    print(f"\n已写入: {output_txt}")