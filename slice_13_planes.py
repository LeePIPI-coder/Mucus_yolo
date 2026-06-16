#!/usr/bin/env python3
"""
将肺部 3D CT 影像按 13 种方向切割，输出 2D 切面图。
所有切割平面均经过粘液栓候选中心点。

重要约定：
    nibabel 读取后的 volume.shape 顺序为 (X, Y, Z)
    所有点坐标、方向向量、法向量均使用 (X, Y, Z) 顺序。

13 种方向分类：
    横矢冠方向：3 种
        axial    : 法向量沿 Z 轴
        coronal  : 法向量沿 Y 轴
        sagittal : 法向量沿 X 轴

    体对角纵切面：4 种
        注意：这里不是“法向量沿体对角线”。
        而是“切面包含体对角线方向”，再通过两个相关棱中点确定唯一纵切面。
        因此 body_diag 的 normal 是纵切面的法向量。

    面对角方向：6 种
        这里仍然按“法向量沿 volume 包围盒面对角线方向”处理。

输出策略：
    - 对每个方向，计算切面与 volume 包围盒的交集范围
    - 以体素索引空间采样
    - 输出无标记图、带中心点图、slices_meta.json
"""

import os
import math
import itertools
import json
import sys
import re

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull


# ═══════════════════════════════════════════════════════════════════════════════
# 调试开关
# ═══════════════════════════════════════════════════════════════════════════════
ENABLE_DIR_FILTER = False        # True=启用 Dir 方向过滤, False=生成全部 13 切面
ENABLE_LOG = False               # True=保存运行日志到文件
FORCE_LONG_EDGE_512 = True     # True=长边始终拉伸到512
LOG_PATH = "/workspace/logs/slice_debug.log"
# ═══════════════════════════════════════════════════════════════════════════════


class Tee:
    """同时写入多个文件对象 stdout + log file。"""

    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# 包围盒 8 个角点，坐标顺序为 (X, Y, Z)
_BOX_CORNERS = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
], dtype=float)

_BOX_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]


def normalize(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("零向量不能归一化")
    return v / norm


def build_normals_from_shape_xyz(shape):
    """
    根据 volume.shape 动态构建 13 个切面法向量。

    约定：
        shape = (X, Y, Z)
        Lx = shape[0] - 1
        Ly = shape[1] - 1
        Lz = shape[2] - 1

    返回：
        normals: [(name, normal), ...]
    """

    Lx = float(shape[0] - 1)
    Ly = float(shape[1] - 1)
    Lz = float(shape[2] - 1)

    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise ValueError(f"非法 volume shape: {shape}")

    normals = [
        # 横矢冠方向，XYZ 顺序
        # axial:    XY 平面，法向量沿 Z
        # coronal:  XZ 平面，法向量沿 Y
        # sagittal: YZ 平面，法向量沿 X
        ("axial",        (0, 0, 1)),
        ("coronal",      (0, 1, 0)),
        ("sagittal",     (1, 0, 0)),

        # 体对角纵切面
        # 这些 normal 不是体对角线方向本身，
        # 而是“包含体对角线方向的纵切面”的法向量。
        #
        # body_diag_0:
        #   体对角方向: ( Lx,  Ly,  Lz)
        #   辅助方向:   (-Lx,  Ly,  0)
        #
        # body_diag_1:
        #   体对角方向: ( Lx,  Ly, -Lz)
        #   辅助方向:   (-Lx,  Ly,  0)
        #
        # body_diag_2:
        #   体对角方向: ( Lx, -Ly,  Lz)
        #   辅助方向:   (-Lx, -Ly,  0)
        #
        # body_diag_3:
        #   体对角方向: (-Lx,  Ly,  Lz)
        #   辅助方向:   ( Lx,  Ly,  0)
        #
        # normal = body_direction × aux_direction
        ("body_diag_0",  (-Ly * Lz, -Lx * Lz,  2 * Lx * Ly)),
        ("body_diag_1",  ( Ly * Lz,  Lx * Lz,  2 * Lx * Ly)),
        ("body_diag_2",  ( Ly * Lz, -Lx * Lz, -2 * Lx * Ly)),
        ("body_diag_3",  (-Ly * Lz,  Lx * Lz, -2 * Lx * Ly)),

        # 面对角方向
        # 这里按照“法向量沿当前 volume 包围盒面对角线方向”处理。
        # 不是简单的 (1,1,0)，而是根据 shape 修正为 (Lx,Ly,0) 等。
        ("face_diag_0",  (Lx,  Ly,  0)),
        ("face_diag_1",  (Lx, -Ly,  0)),
        ("face_diag_2",  (Lx,  0,  Lz)),
        ("face_diag_3",  (Lx,  0, -Lz)),
        ("face_diag_4",  (0,   Ly,  Lz)),
        ("face_diag_5",  (0,   Ly, -Lz)),
    ]

    return normals


def unique_points(points, tol=1e-8):
    """点去重。"""
    uniq = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if not any(np.linalg.norm(p - q) < tol for q in uniq):
            uniq.append(p)
    return uniq


def orthonormal_basis(n):
    """
    由法向量 n 构造平面的一组正交基 u, v。
    坐标顺序为 (X, Y, Z)。
    """
    n = normalize(n)

    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    u = np.cross(n, ref)
    u = normalize(u)

    v = np.cross(n, u)
    v = normalize(v)

    return u, v


def plane_bbox_intersection(shape, center, n, u, v):
    """
    计算平面 n·(p-center)=0 与 volume 包围盒的交集范围。

    shape:
        volume.shape, 顺序为 (X, Y, Z)

    center:
        切割中心，顺序为 (X, Y, Z)

    返回：
        u_min, u_max, v_min, v_max
    """

    max_corner = np.array(shape, dtype=float) - 1.0
    corners = _BOX_CORNERS * max_corner

    n = normalize(n)
    # 保存切面与每个棱的交点
    dists = (corners - center) @ n

    pts = []
    for i0, i1 in _BOX_EDGES:
        d0, d1 = dists[i0], dists[i1]

        if d0 * d1 > 0:
            continue

        if abs(d0 - d1) < 1e-12:
            continue

        t = d0 / (d0 - d1)
        p = corners[i0] + t * (corners[i1] - corners[i0])

        dp = p - center
        pts.append([dp @ u, dp @ v])

    if len(pts) < 3:
        diag = np.linalg.norm(max_corner) / 2.0
        return -diag, diag, -diag, diag

    pts = np.array(pts)

    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]

    u_min, u_max = hull_pts[:, 0].min(), hull_pts[:, 0].max()
    v_min, v_max = hull_pts[:, 1].min(), hull_pts[:, 1].max()

    return u_min, u_max, v_min, v_max


def sample_plane(
    volume,
    center,
    normal,
    n_samples=512,
    margin_vox=0,
    fill_value=None,
    force_long_edge=False,
):
    """
    从 3D volume 中采样一个经过 center、法向为 normal 的平面。

    volume:
        3D numpy array, shape = (X, Y, Z)

    center:
        (X, Y, Z)

    normal:
        平面法向量，顺序为 (X, Y, Z)

    返回：
        slice_2d, meta
    """

    normal = normalize(normal)
    u, v = orthonormal_basis(normal)

    shape = np.array(volume.shape)

    u_min, u_max, v_min, v_max = plane_bbox_intersection(
        shape, center, normal, u, v
    )

    u_min -= margin_vox
    u_max += margin_vox
    v_min -= margin_vox
    v_max += margin_vox

    u_extent = u_max - u_min
    v_extent = v_max - v_min

    if force_long_edge:
        if u_extent >= v_extent:
            nu = n_samples
            nv = max(2, int(np.round(n_samples * v_extent / u_extent)))
        else:
            nv = n_samples
            nu = max(2, int(np.round(n_samples * u_extent / v_extent)))
    else:
        u_px = max(2, int(np.round(u_extent)))
        v_px = max(2, int(np.round(v_extent)))

        if max(u_px, v_px) <= n_samples:
            nu, nv = u_px, v_px
        else:
            if u_extent >= v_extent:
                nu = n_samples
                nv = max(2, int(np.round(n_samples * v_extent / u_extent)))
            else:
                nv = n_samples
                nu = max(2, int(np.round(n_samples * u_extent / v_extent)))

    u_lin = np.linspace(u_min, u_max, nu)
    v_lin = np.linspace(v_min, v_max, nv)

    U, V = np.meshgrid(u_lin, v_lin, indexing="ij")

    pts = (
        center[:, None, None]
        + U[None, ...] * u[:, None, None]
        + V[None, ...] * v[:, None, None]
    )

    if fill_value is None:
        fill_value = float((volume.min() + volume.max()) / 2)

    coords = pts.reshape(3, -1)

    sampled = map_coordinates(
        volume,
        coords,
        order=0,
        mode="constant",
        cval=fill_value,
    )

    content = sampled.reshape(nu, nv)

    cx_content = (0.0 - u_min) / u_extent * (nu - 1) if u_extent > 0 else (nu - 1) / 2
    cy_content = (0.0 - v_min) / v_extent * (nv - 1) if v_extent > 0 else (nv - 1) / 2

    target = n_samples
    padded = np.full((target, target), fill_value, dtype=content.dtype)

    u_start = (target - nu) // 2
    v_start = (target - nv) // 2

    padded[u_start:u_start + nu, v_start:v_start + nv] = content

    center_col = cx_content + u_start
    center_row = cy_content + v_start

    meta = {
        "width_px": target,
        "height_px": target,
        "content_width_px": int(nu),
        "content_height_px": int(nv),
        "u_extent_vx": round(float(u_extent), 1),
        "v_extent_vx": round(float(v_extent), 1),
        "aspect": round(float(nu / nv), 3) if nv > 0 else 1.0,
        "center_col": round(float(center_col), 1),
        "center_row": round(float(center_row), 1),
        "normal": [round(float(x), 6) for x in normal],
    }

    return padded, meta


def parse_annotation_file(filepath):
    """
    解析标注文件，提取 [Mask] 段中的 mask 名称和坐标。

    文件格式示例：
        [Mask]
        Mask1 53.0 82.0 239.0

    坐标顺序：
        nibabel 当前约定为 (X, Y, Z)

    返回：
        [(mask_name, center_xyz), ...]
    """

    masks = []
    in_mask_section = False

    if not filepath or not os.path.exists(filepath):
        return masks

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line == "[Mask]":
                in_mask_section = True
                continue

            if line.startswith("[") and line != "[Mask]":
                in_mask_section = False
                continue

            if in_mask_section:
                parts = line.split()
                if len(parts) >= 4:
                    mask_name = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    center = np.array([x, y, z], dtype=float)
                    masks.append((mask_name, center))

    return masks


def parse_dir_vectors(filepath):
    """
    解析标注文件，提取 [XDir] 段中的方向向量。

    文件格式示例：
        [1Dir] 43.6 79.2 231.8 | 65.6 87.6 250.6

    坐标顺序：
        (X, Y, Z)

    返回：
        {mask_index: direction_vector_xyz}
    """

    dir_vectors = {}

    if not filepath or not os.path.exists(filepath):
        return dir_vectors

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            m = re.match(
                r"\[(\d+)Dir\]\s+"
                r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|\s*"
                r"([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                line,
            )

            if m:
                idx = int(m.group(1))

                x1, y1, z1 = float(m.group(2)), float(m.group(3)), float(m.group(4))
                x2, y2, z2 = float(m.group(5)), float(m.group(6)), float(m.group(7))

                start = np.array([x1, y1, z1], dtype=float)
                end = np.array([x2, y2, z2], dtype=float)

                vec = end - start

                if np.linalg.norm(vec) > 1e-6:
                    dir_vectors[idx] = vec

    return dir_vectors


def angle_between(v1, v2):
    """
    两个向量夹角，范围 [0, 180] 度。
    """

    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-12 or n2 < 1e-12:
        return 180.0

    cos = np.dot(v1, v2) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos)))


def process_center(
    volume,
    center,
    outdir,
    outdir_label,
    args,
    normals,
    dir_vector=None,
):
    """
    针对单个切割中心，生成并保存 13 个方向的切面。

    每个方向生成 3 层：
        center
        center + 1 voxel along normal
        center - 1 voxel along normal
    """

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_label, exist_ok=True)

    active_normals = []

    for name, normal in normals:
        normal = np.asarray(normal, dtype=float)

        if dir_vector is not None:
            ang = angle_between(normal, dir_vector)

            # 只保留法向量与 Dir 近似垂直的切面
            # 即切面近似包含 Dir 方向
            if ang < 80.0 or ang > 100.0:
                print(f"    跳过 {name:20s}  夹角={ang:.1f}°")
                continue

        active_normals.append((name, normal))

    print(f"    使用 {len(active_normals)}/{len(normals)} 个切面方向")

    all_meta = {}

    for name, normal in active_normals:
        n_unit = normalize(normal)

        offsets = [
            (0, ""),
            (1, "_p1"),
            (-1, "_m1"),
        ]

        for offset, suffix in offsets:
            shifted_center = center + offset * n_unit

            fv = 0.0
            if args.vmin is not None and args.vmax is not None:
                fv = (args.vmin + args.vmax) / 2.0

            slice_2d, meta = sample_plane(
                volume,
                shifted_center,
                normal,
                n_samples=args.n_samples,
                margin_vox=args.margin,
                fill_value=fv,
                force_long_edge=args.force_512,
            )

            meta["offset_vx"] = offset

            key = f"{name}{suffix}"
            all_meta[key] = meta

            cx, cy = meta["center_col"], meta["center_row"]
            content_w = meta["content_width_px"]
            content_h = meta["content_height_px"]

            offset_str = f"({offset:+d})" if offset != 0 else "(0)"

            print(
                f"    {key:24s} {offset_str:4s} "
                f"content={content_w:4d}×{content_h:4d}px  "
                f"extent=({meta['u_extent_vx']:6.1f}, {meta['v_extent_vx']:6.1f})vx  "
                f"center=({cx:.0f},{cy:.0f})px  "
                f"aspect={meta['aspect']:.3f}"
            )

            if args.no_png:
                continue

            out_path = os.path.join(outdir, f"{key}.png")
            plt.imsave(
                out_path,
                slice_2d.T,
                cmap=args.cmap,
                origin="lower",
                vmin=args.vmin,
                vmax=args.vmax,
            )

            dpi = 100
            fig, ax = plt.subplots(
                figsize=(meta["width_px"] / dpi, meta["height_px"] / dpi),
                facecolor="gray",
            )
            ax.set_facecolor("gray")

            ax.imshow(
                slice_2d.T,
                cmap=args.cmap,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
                vmin=args.vmin,
                vmax=args.vmax,
            )

            ax.plot(cx, cy, "ro", markersize=2, markeredgewidth=0)
            ax.axis("off")

            fig.tight_layout(pad=0)

            marked_path = os.path.join(outdir_label, f"{key}.png")
            fig.savefig(marked_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    meta_path = os.path.join(outdir, "slices_meta.json")

    with open(meta_path, "w") as f:
        json.dump(
            {
                "center_voxel_xyz": [round(float(c), 1) for c in center],
                "volume_shape_xyz": list(volume.shape),
                "offsets": {"": 0, "_p1": 1, "_m1": -1},
                "slices": all_meta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"    元信息已保存到: {meta_path}")

    return all_meta


def main():
    import argparse

    parser = argparse.ArgumentParser(description="按13方向切割3D影像，XYZ坐标顺序")

    parser.add_argument(
        "input",
        nargs="?",
        default="/data/Mucus_data/E0001425_20100927/lung_image.nii.gz",
        help="输入 .nii.gz 路径",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        default="./data/rotate_slice_260615/slice",
        help="无标记切面图输出目录",
    )

    parser.add_argument(
        "-ol",
        "--outdir_label",
        default="./data/rotate_slice_260615/slice_label",
        help="带中心点标记切面图输出目录",
    )

    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=512,
        help="输出图像尺寸，默认 512",
    )

    parser.add_argument(
        "-m",
        "--margin",
        type=int,
        default=0,
        help="包围盒外扩体素数，默认 0",
    )

    parser.add_argument(
        "--cmap",
        default="gray",
        help="colormap，默认 gray",
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=-1450,
        help="窗宽窗位下限 HU",
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=50,
        help="窗宽窗位上限 HU",
    )

    parser.add_argument(
        "--center",
        default=[53.0, 82.0, 239.0],
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="切割中心点，体素坐标 X Y Z，与 --anno 互斥",
    )

    parser.add_argument(
        "--anno",
        default="/data/Mucus_data/E0001425_20100927/annotation_points_cropped.txt",
        help="标注点文件路径，读取 [Mask] 段批量切割",
    )

    parser.add_argument(
        "--force-512",
        action="store_true",
        default=FORCE_LONG_EDGE_512,
        help="强制长边始终拉伸到 n_samples",
    )

    parser.add_argument(
        "--no-png",
        action="store_true",
        help="只输出元信息，不生成图片",
    )

    parser.add_argument(
        "--no-dir-filter",
        action="store_true",
        help="禁用 Dir 方向角度过滤，生成全部 13 个切面",
    )

    parser.add_argument(
        "--log",
        default=None,
        help="将运行日志保存到指定文件",
    )

    args = parser.parse_args()

    log_fh = None
    original_stdout = sys.stdout

    if args.log:
        log_fh = open(args.log, "w")
        sys.stdout = Tee(original_stdout, log_fh)
    elif ENABLE_LOG:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log_fh = open(LOG_PATH, "w")
        sys.stdout = Tee(original_stdout, log_fh)
        print(f"[代码开关] 日志已启用，保存到: {LOG_PATH}")

    nii = nib.load(args.input)
    volume = nii.get_fdata().astype(np.float32)

    print(f"影像尺寸 XYZ: {volume.shape}")

    normals = build_normals_from_shape_xyz(volume.shape)

    print("\n动态生成的 13 个切面法向量：")
    for name, normal in normals:
        n = normalize(normal)
        print(f"  {name:16s} raw={normal}  unit={tuple(np.round(n, 4))}")

    mask_points = parse_annotation_file(args.anno)
    dir_vectors = parse_dir_vectors(args.anno)

    print(
        f"\n从标注文件读取到 {len(mask_points)} 个 mask 点: "
        f"{', '.join(m[0] for m in mask_points)}"
    )

    if dir_vectors:
        print(f"从标注文件读取到 {len(dir_vectors)} 个 Dir 方向向量")

    if not mask_points:
        center = np.array(args.center, dtype=float)

        print(
            f"\n中心点 XYZ: "
            f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
        )

        process_center(
            volume,
            center,
            args.outdir,
            args.outdir_label,
            args,
            normals=normals,
        )

    else:
        for mask_name, center in mask_points:
            mask_idx = int(mask_name.replace("Mask", ""))

            if args.no_dir_filter or not ENABLE_DIR_FILTER:
                dir_vector = None
            else:
                dir_vector = dir_vectors.get(mask_idx)

            print(f"\n{'=' * 60}")
            print(
                f"处理 {mask_name}: center XYZ="
                f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
            )

            if dir_vector is not None:
                print(
                    f"  Dir 向量 XYZ: "
                    f"({dir_vector[0]:.1f}, {dir_vector[1]:.1f}, {dir_vector[2]:.1f})"
                )

            print(f"{'=' * 60}")

            outdir = os.path.join(args.outdir, mask_name)
            outdir_label = os.path.join(args.outdir_label, mask_name)

            process_center(
                volume,
                center,
                outdir,
                outdir_label,
                args,
                normals=normals,
                dir_vector=dir_vector,
            )

    print(f"\n完成。无标记切面图: {args.outdir}/")
    print(f"带中心标记切面图: {args.outdir_label}/")

    if log_fh:
        sys.stdout = original_stdout
        log_fh.close()


if __name__ == "__main__":
    main()