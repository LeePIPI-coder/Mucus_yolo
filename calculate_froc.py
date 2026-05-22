import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
import cc3d
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist


def load_mask(mask_path):
    """
    加载nii.gz格式的掩码文件
    
    Args:
        mask_path: 掩码文件路径
    
    Returns:
        mask_array: 掩码的numpy数组
        info: 掩码的元数据信息
    """
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    info = {
        'origin': mask.GetOrigin(),
        'spacing': mask.GetSpacing(),
        'direction': mask.GetDirection()
    }
    return mask_array, info


def load_gt_boxes_from_csv(csv_path, fold=None):
    """
    从GT_bboxes.csv直接读取GT边界框（体素坐标），跳过掩码加载和CC3D分析。

    CSV列: voxel_center_i(宽度/x), voxel_center_j(高度/y), voxel_center_k(深度/z),
           voxel_dim_i, voxel_dim_j, voxel_dim_k

    Returns:
        gt_boxes_dict: {patient_key: [[z_min, y_min, x_min, z_max, y_max, x_max], ...]}
    """
    df = pd.read_csv(csv_path)
    if fold is not None:
        df = df[df['fold'] == fold]

    gt_boxes_dict = defaultdict(list)
    for _, row in df.iterrows():
        patient = row['patient_key']
        ci, cj, ck = row['voxel_center_i'], row['voxel_center_j'], row['voxel_center_k']
        di, dj, dk = row['voxel_dim_i'], row['voxel_dim_j'], row['voxel_dim_k']

        x_min = ci - di / 2
        x_max = ci + di / 2
        y_min = cj - dj / 2
        y_max = cj + dj / 2
        z_min = ck - dk / 2
        z_max = ck + dk / 2

        gt_boxes_dict[patient].append([z_min, y_min, x_min, z_max, y_max, x_max])

    return gt_boxes_dict


def get_mask_metadata(mask_path):
    """只加载掩码文件的元数据（原点、间距、方向），不读取像素数据。"""
    mask = sitk.ReadImage(str(mask_path))
    return {
        'origin': mask.GetOrigin(),
        'spacing': mask.GetSpacing(),
        'direction': mask.GetDirection()
    }


def mask_to_3d_boxes(mask_array):
    """
    从3D掩码计算3D边界框
    
    Args:
        mask_array: 3D掩码数组 (depth, height, width)
    
    Returns:
        boxes_3d: 3D边界框列表 [(z_min, y_min, x_min, z_max, y_max, x_max)]
    """
    # 将掩码转换为二值图像
    binary_mask = (mask_array > 0).astype(np.uint8)
    
    # 连通组件标记
    labels_out = cc3d.connected_components(binary_mask, connectivity=26)  # 3D连通性
    num_objects = labels_out.max()
    
    boxes_3d = []
    
    if num_objects == 0:
        return boxes_3d
    
    for obj_label in range(1, num_objects + 1):
        obj_mask = (labels_out == obj_label).astype(np.uint8)
        
        # 获取对象的非零坐标
        coords = np.where(obj_mask)
        if len(coords[0]) == 0:  # 如果没有找到非零元素
            continue
        
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        # 添加一些边距
        margin = 5
        z_min = max(z_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_min = max(x_min - margin, 0)
        z_max = min(z_max + margin, mask_array.shape[0] - 1)
        y_max = min(y_max + margin, mask_array.shape[1] - 1)
        x_max = min(x_max + margin, mask_array.shape[2] - 1)
        
        boxes_3d.append((z_min, y_min, x_min, z_max, y_max, x_max))
    
    return boxes_3d


def calculate_3d_iou(box1, box2):
    """
    计算两个3D边界框的IoU
    
    Args:
        box1: 第一个3D边界框 [z1_min, y1_min, x1_min, z1_max, y1_max, x1_max]
        box2: 第二个3D边界框 [z2_min, y2_min, x2_min, z2_max, y2_max, x2_max]
    
    Returns:
        iou: 3D交并比
    """
    z1_min, y1_min, x1_min, z1_max, y1_max, x1_max = box1
    z2_min, y2_min, x2_min, z2_max, y2_max, x2_max = box2
    
    # 计算交集的边界
    zi_min = max(z1_min, z2_min)
    yi_min = max(y1_min, y2_min)
    xi_min = max(x1_min, x2_min)
    zi_max = min(z1_max, z2_max)
    yi_max = min(y1_max, y2_max)
    xi_max = min(x1_max, x2_max)
    
    # 计算交集体积
    inter_depth = max(0, zi_max - zi_min)
    inter_height = max(0, yi_max - yi_min)
    inter_width = max(0, xi_max - xi_min)
    inter_volume = inter_depth * inter_height * inter_width
    
    # 计算各自体积
    vol1 = (z1_max - z1_min) * (y1_max - y1_min) * (x1_max - x1_min)
    vol2 = (z2_max - z2_min) * (y2_max - y2_min) * (x2_max - x2_min)
    
    # 计算并集体积
    union_volume = vol1 + vol2 - inter_volume
    
    return inter_volume / union_volume if union_volume > 0 else 0


def calculate_froc_3d(predictions, ground_truths_3d, patient_count, iou_threshold):
    """
    使用3D边界框计算FROC曲线
    
    Args:
        predictions: 预测结果列表，每个元素为 (score, [z_min, y_min, x_min, z_max, y_max, x_max])
        ground_truths_3d: 3D ground truth边界框列表 [(z_min, y_min, x_min, z_max, y_max, x_max)]
        patient_count: 病例总数
        iou_threshold: 3D IoU阈值
    
    Returns:
        fps: 假阳性率
        sensitivities: 灵敏度
        thresholds: 对应的置信度阈值
    """
    # 按置信度排序预测结果
    predictions.sort(key=lambda x: x[0], reverse=True)
    
    # 计算ground truth总数
    total_gt = len(ground_truths_3d)
    if total_gt == 0:
        return [], [], []
    
    # 跟踪每个ground truth是否被检测到
    detected = [False] * len(ground_truths_3d)
    
    # 计算每个阈值下的TP和FP
    tp = 0
    fp = 0
    fps = []
    sensitivities = []
    thresholds = []  # 记录每个点的阈值
    
    for pred in predictions:
        # 支持2元素元组 (score, pred_box) 或 3元素元组 (score, pred_box, row)
        if len(pred) >= 2:
            score, pred_box = pred[0], pred[1]
        else:
            continue
        
        # 找到IoU最大的ground truth
        max_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(ground_truths_3d):
            if not detected[i]:
                pred_box_3d = pred_box
                
                iou = calculate_3d_iou(pred_box_3d, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = i
        
        if max_iou >= iou_threshold:
            # 真正例
            tp += 1
            detected[best_gt_idx] = True
        else:
            # 假正例
            fp += 1
        
        # 计算灵敏度和假阳性率
        sensitivity = tp / total_gt
        avg_fp_per_patient = fp / patient_count  # 平均每个病例的假阳性数
        
        fps.append(avg_fp_per_patient)
        sensitivities.append(sensitivity)
        thresholds.append(score)  # 当前使用的置信度阈值
    
    return fps, sensitivities, thresholds


def plot_froc_curve(fps, sensitivities, fold=0, save_path="/workspace/resutls_froc"):
    """
    绘制FROC曲线
    
    Args:
        fps: 假阳性率列表
        sensitivities: 灵敏度列表
        fold: 折数
        save_path: 保存路径，默认"/workspace/resutls_froc"
    """
    if not fps or not sensitivities:
        print("没有FROC数据可绘制")
        return
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制FROC曲线
    plt.plot(fps, sensitivities, 'b-', linewidth=2, label='FROC Curve')
    
    # 添加标题和标签
    plt.title(f'FROC Curve - Fold {fold}', fontsize=16)
    plt.xlabel('Average Number of False Positives Per Patient', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 设置坐标轴范围
    plt.xlim(0, max(fps) if fps else 1)
    plt.ylim(0, 1)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 保存图片
    plot_path = f"{save_path}/froc_curve_fold_{fold}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"FROC曲线图已保存到 {plot_path}")
    
    # 显示图表
    plt.show()


def find_sensitivity_at_target_fp(fps, sensitivities, thresholds, target_fp_per_patient):
    """
    计算平均每个病例出现目标FP数时对应的sensitivity和置信度阈值
    
    Args:
        fps: 假阳性率列表 (平均每个病例的FP数)
        sensitivities: 灵敏度列表
        thresholds: 对应的置信度阈值列表
        target_fp_per_patient: 目标平均每个病例的FP数
    
    Returns:
        sensitivity_at_target_fp: 目标FP数下的sensitivity
        threshold_at_target_fp: 目标FP数下的置信度阈值
    """
    if not fps:
        return 0, 0

    # 查找最接近目标FP数的点
    # 在FROC计算中，fps是累积的FP数除以病人数，所以我们要找到最接近目标值的点
    diffs = [abs(fp - target_fp_per_patient) for fp in fps]
    closest_idx = diffs.index(min(diffs))
    
    sensitivity_at_target_fp = sensitivities[closest_idx]
    threshold_at_target_fp = thresholds[closest_idx] if closest_idx < len(thresholds) else 0
    
    return sensitivity_at_target_fp, threshold_at_target_fp


def coord_pat2vox(pat, origin, spacing, direction):
    """
    将世界坐标转换为像素坐标
    
    Args:
        pat: 世界坐标 [x, y, z]
        origin: 图像原点
        spacing: 像素间距
        direction: 方向矩阵
    
    Returns:
        voxel_coord: 像素坐标 (x, y, z)
    """
    origin = np.array(origin)
    spacing = np.array(spacing)
    direction = np.array(direction)
    direction_matrix = direction.reshape(3, 3)
    transformation_matrix = direction_matrix * spacing
    pat = np.array(pat)
    voxel_coord = np.linalg.inv(transformation_matrix).dot(pat - origin)
    return tuple(voxel_coord)


def match_predictions_with_labels(predictions_with_details, ground_truths_3d, patient_count, iou_threshold=0.01, min_confidence=None):
    """
    将预测结果与标签进行匹配，创建带有TP/FP标签的新表格

    Args:
        predictions_with_details: 预测结果列表，每个元素为 (score, pred_box, original_row_data)
        ground_truths_3d: 3D ground truth边界框列表 [(z_min, y_min, x_min, z_max, y_max, x_max)]
        patient_count: 病例总数
        iou_threshold: 3D IoU阈值
        min_confidence: 最小置信度阈值，低于此值的预测将被跳过

    Returns:
        results_df: 包含预测信息和TP/FP标签的DataFrame
    """
    if not predictions_with_details:
        return pd.DataFrame()

    # 按置信度排序预测结果
    predictions_sorted = sorted(predictions_with_details, key=lambda x: x[0], reverse=True)

    # 跟踪每个ground truth是否被检测到
    detected = [False] * len(ground_truths_3d)

    results = []

    for pred in predictions_sorted:
        score, pred_box, original_row = pred

        # 跳过低于最小置信度阈值的预测
        if min_confidence is not None and score < min_confidence:
            continue

        # 找到IoU最大的ground truth
        max_iou = 0
        best_gt_idx = -1

        for i, gt_box in enumerate(ground_truths_3d):
            if not detected[i]:
                iou = calculate_3d_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = i

        if max_iou >= iou_threshold:
            # 真正例
            label = "TP"
            detected[best_gt_idx] = True
        else:
            # 假正例
            label = "FP"

        # 将原始行数据转换为字典
        row_dict = original_row.to_dict() if hasattr(original_row, 'to_dict') else dict(original_row)

        # 添加预测框的像素坐标信息
        z_min, y_min, x_min, z_max, y_max, x_max = pred_box
        row_dict['pixel_x_min'] = x_min
        row_dict['pixel_y_min'] = y_min
        row_dict['pixel_z_min'] = z_min
        row_dict['pixel_x_max'] = x_max
        row_dict['pixel_y_max'] = y_max
        row_dict['pixel_z_max'] = z_max
        row_dict['pixel_center_x'] = (x_min + x_max) / 2
        row_dict['pixel_center_y'] = (y_min + y_max) / 2
        row_dict['pixel_center_z'] = (z_min + z_max) / 2
        row_dict['pixel_width'] = x_max - x_min
        row_dict['pixel_height'] = y_max - y_min
        row_dict['pixel_depth'] = z_max - z_min

        # 添加IoU值和标签
        row_dict['iou'] = max_iou
        row_dict['prediction_type'] = label

        results.append(row_dict)

    results_df = pd.DataFrame(results)

    # 调整列顺序，将prediction_type放在最后
    if not results_df.empty:
        cols = [col for col in results_df.columns if col != 'prediction_type'] + ['prediction_type']
        results_df = results_df[cols]

    return results_df


def process_fold_predictions(fold=0, save_path="/workspace/all_results/resutls_froc", iou_threshold=0.01, min_confidence=None, gt_csv_path=None):
    """
    处理指定折的预测结果，使用3D边界框计算FROC曲线

    Args:
        fold: 折数
        save_path: 结果保存路径
        gt_csv_path: GT_bboxes.csv路径，如果提供则直接从CSV读取GT框（跳过掩码CC3D分析）
    """
    # 读取预测结果
    pred_csv_path = f"/workspace/all_results/predictions_by_fold/249_neg_0/predictions_fold_{fold}.csv"
    if not os.path.exists(pred_csv_path):
        print(f"预测结果文件不存在: {pred_csv_path}")
        return

    pred_df = pd.read_csv(pred_csv_path)
    print(f"读取到 {len(pred_df)} 条预测结果")

    # 如果提供了GT CSV，直接从CSV加载GT框
    gt_boxes_by_patient = None
    if gt_csv_path is not None and os.path.exists(gt_csv_path):
        print(f"从CSV加载GT框: {gt_csv_path}")
        gt_boxes_by_patient = load_gt_boxes_from_csv(gt_csv_path, fold=fold)
        print(f"从CSV加载了 {sum(len(v) for v in gt_boxes_by_patient.values())} 个GT框，涉及 {len(gt_boxes_by_patient)} 个患者")

    # 按患者和路径分组处理
    grouped = pred_df.groupby(['patient_key', 'path'])
    print(f"共有 {len(grouped)} 个数据样本")

    all_predictions_with_details = []  # 包含详细信息的预测列表
    all_ground_truths_3d = []  # 存储所有3D ground truth
    unique_patients = set()  # 用于统计病例数
    processed_masks = set()  # 避免重复处理相同的掩码
    mask_metadata_cache = {}  # 缓存掩码元数据

    for (patient, image_path), group in grouped:
        print(f"处理患者: {patient}")
        print(f"图像路径: {image_path}")

        # 添加到唯一病例集合
        unique_patients.add(patient)

        # 根据图像路径找到掩码路径（将"image"替换为"mask"）
        image_path = Path(image_path)
        if "DICOM" not in str(image_path):
            mask_dir = image_path.parent.parent

        else:
            mask_dir = image_path.parent.parent.parent

        mask_path = next(mask_dir.glob("*.nii.gz"), None)

        print(f"掩码路径: {mask_path}")

        if mask_path is None or not mask_path.exists():
            print(f"掩码文件不存在: {mask_path}")
            continue

        # 获取掩码元数据（用于坐标转换），缓存避免重复读取
        mask_path_str = str(mask_path)
        if mask_path_str not in mask_metadata_cache:
            mask_metadata_cache[mask_path_str] = get_mask_metadata(mask_path_str)
        mask_info = mask_metadata_cache[mask_path_str]

        # GT框来源：优先使用CSV，否则从掩码CC3D分析
        if gt_boxes_by_patient is not None:
            # 从CSV获取该患者的GT框
            if patient in gt_boxes_by_patient and mask_path_str not in processed_masks:
                all_ground_truths_3d.extend(gt_boxes_by_patient[patient])
                processed_masks.add(mask_path_str)
        else:
            # 回退：从掩码CC3D分析获取GT框
            if mask_path_str not in processed_masks:
                mask_array, mask_info = load_mask(str(mask_path))
                gt_boxes_3d = mask_to_3d_boxes(mask_array)
                all_ground_truths_3d.extend(gt_boxes_3d)
                processed_masks.add(mask_path_str)

        # 收集预测结果（带详细信息）
        for idx, row in group.iterrows():
            score = row['detector_score']

            # 计算边界框的像素坐标
            min_x = row['roi_patientPos_min_x']
            min_y = row['roi_patientPos_min_y']
            min_z = row['roi_patientPos_min_z']
            max_x = row['roi_patientPos_max_x']
            max_y = row['roi_patientPos_max_y']
            max_z = row['roi_patientPos_max_z']

            # 将边界框坐标转换为像素坐标
            min_vox = coord_pat2vox(
                [min_x, min_y, min_z],
                mask_info['origin'],
                mask_info['spacing'],
                mask_info['direction']
            )
            max_vox = coord_pat2vox(
                [max_x, max_y, max_z],
                mask_info['origin'],
                mask_info['spacing'],
                mask_info['direction']
            )

            x1 = min_vox[0]  # x坐标
            y1 = min_vox[1]  # y坐标
            z1 = min_vox[2]  # z坐标
            x2 = max_vox[0]  # x坐标
            y2 = max_vox[1]  # y坐标
            z2 = max_vox[2]  # z坐标

            # 创建3D边界框 [z_min, y_min, x_min, z_max, y_max, x_max]
            pred_box = [z1, y1, x1, z2, y2, x2]

            # 保存预测结果，包含原始行数据用于后续输出
            all_predictions_with_details.append((score, pred_box, row))

    # 计算病例数
    patient_count = len(unique_patients)
    print(f"病例总数: {patient_count}")
    print(f"3D Ground Truth 数量: {len(all_ground_truths_3d)}")
    print(f"预测数量: {len(all_predictions_with_details)}")

    # 创建带有TP/FP标签的结果表格
    results_df = match_predictions_with_labels(
        all_predictions_with_details, all_ground_truths_3d, patient_count, iou_threshold, min_confidence
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保存带有TP/FP标签的结果
    if not results_df.empty:
        results_output_path = f"{save_path}/Prediction_TP_FP_fold_{fold}.csv"
        results_df.to_csv(results_output_path, index=False)
        print(f"带有TP/FP标签的预测结果已保存到 {results_output_path}")
        print(f"TP数量: {(results_df['prediction_type'] == 'TP').sum()}")
        print(f"FP数量: {(results_df['prediction_type'] == 'FP').sum()}")

    # 计算FROC曲线
    fps, sensitivities, thresholds = calculate_froc_3d(all_predictions_with_details, all_ground_truths_3d, patient_count, iou_threshold)
    
    # 保存结果
    froc_df = pd.DataFrame({
        'fps': fps,
        'sensitivity': sensitivities,
        'thresholds': thresholds
    })
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    froc_df.to_csv(f"{save_path}/froc_fold_{fold}.csv", index=False)
    print(f"FROC曲线数据已保存到 {save_path}/froc_fold_{fold}.csv")
    
    # 绘制FROC曲线
    plot_froc_curve(fps, sensitivities, fold, save_path)
    
    # 计算平均每个病例出现100个FP时对应的置信度阈值和sensitivity
    target_fp_per_patient = 100
    target_sensitivity, target_threshold = find_sensitivity_at_target_fp(fps, sensitivities, thresholds, target_fp_per_patient)
    
    # 打印结果
    print("FROC曲线计算完成")
    print(f"最大灵敏度: {max(sensitivities) if sensitivities else 0:.4f}")
    print(f"对应的假阳性数: {fps[sensitivities.index(max(sensitivities))] if sensitivities else 0:.4f}")
    print(f"平均每个病例出现{target_fp_per_patient}个FP时的灵敏度: {target_sensitivity:.4f}")
    print(f"平均每个病例出现{target_fp_per_patient}个FP时的置信度阈值: {target_threshold:.4f}")


if __name__ == "__main__":
    save_path = "/workspace/all_results/results_froc/249_neg_0_260520"
    # 处理第1折的预测结果
    # 设置min_confidence过滤低置信度预测，例如0.5
    process_fold_predictions(fold=4, save_path=save_path, iou_threshold=0.001, min_confidence=None)