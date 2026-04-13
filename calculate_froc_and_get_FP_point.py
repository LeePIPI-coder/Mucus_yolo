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


def calculate_froc_3d_with_fp_details(predictions_with_details, ground_truths_3d, patient_count, iou_threshold=0.01):
    """
    使用3D边界框计算FROC曲线，同时记录FP详情
    
    Args:
        predictions_with_details: 包含详细信息的预测结果列表，每个元素为 (score, pred_box, original_row_data)
        ground_truths_3d: 3D ground truth边界框列表 [(z_min, y_min, x_min, z_max, y_max, x_max)]
        patient_count: 病例总数
        iou_threshold: 3D IoU阈值
    
    Returns:
        fps: 假阳性率
        sensitivities: 灵敏度
        thresholds: 对应的置信度阈值
        fp_details: FP点的详细信息列表
    """
    # 按置信度排序预测结果
    predictions_with_details.sort(key=lambda x: x[0], reverse=True)
    
    # 计算ground truth总数
    total_gt = len(ground_truths_3d)
    if total_gt == 0:
        return [], [], [], []
    
    # 跟踪每个ground truth是否被检测到
    detected = [False] * len(ground_truths_3d)
    
    # 计算每个阈值下的TP和FP
    tp = 0
    fp = 0
    fps = []
    sensitivities = []
    thresholds = []  # 记录每个点的阈值
    fp_details = []  # 记录FP的详细信息
    
    for pred in predictions_with_details:
        score, pred_box, original_row = pred
        
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
            # if best_gt_idx != -1:
            detected[best_gt_idx] = True
        else:
            # 假正例
            fp += 1
            fp_details.append(original_row)  # 记录FP的原始行数据
        
        # 计算灵敏度和假阳性率
        sensitivity = tp / total_gt if total_gt > 0 else 0
        avg_fp_per_patient = fp / patient_count  # 平均每个病例的假阳性数
        
        fps.append(avg_fp_per_patient)
        sensitivities.append(sensitivity)
        thresholds.append(score)  # 当前使用的置信度阈值
    
    return fps, sensitivities, thresholds, fp_details


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


def calculate_froc_auc(fps, sensitivities, fixed_fp_points=None):
    """
    计算FROC曲线下面积 (FROC-AUC)

    Args:
        fps: 假阳性率列表 (平均每个病例的FP数)
        sensitivities: 灵敏度列表
        fixed_fp_points: 固定的FP数列表，如 [1, 5, 10, 20]。
                        如果为None，则使用整个FPS范围计算AUC。
                        如果指定，则只在这些固定点范围内计算AUC。

    Returns:
        auc: FROC曲线下面积
    """
    if not fps or not sensitivities or len(fps) != len(sensitivities):
        return 0.0

    # 对fps排序（确保x轴递增）
    sorted_pairs = sorted(zip(fps, sensitivities))
    sorted_fps = [p[0] for p in sorted_pairs]
    sorted_sens = [p[1] for p in sorted_pairs]

    if fixed_fp_points is None:
        # 原始逻辑：使用整个FPS范围
        auc = np.trapz(sorted_sens, sorted_fps)
    else:
        # 固定FP点数计算：在指定的FP点范围内计算AUC
        max_fp = max(sorted_fps) if sorted_fps else 0
        min_fp = 0.0

        # 过滤有效的固定点（不超过最大FP数）
        valid_fp_points = sorted([fp for fp in fixed_fp_points if fp <= max_fp])
        if not valid_fp_points:
            # 如果没有有效点，使用整个范围
            auc = np.trapz(sorted_sens, sorted_fps)
        else:
            # 使用线性插值获取每个固定FP点对应的sensitivity
            # fps可能不是从0开始，需要补充(0, 0)点
            interp_fps = [0.0] + sorted_fps
            interp_sens = [0.0] + sorted_sens

            # 创建插值函数
            from scipy.interpolate import interp1d
            interp_func = interp1d(interp_fps, interp_sens, kind='linear',
                                  bounds_error=False, fill_value=(0.0, sorted_sens[-1]))

            # 获取固定点对应的sensitivity
            sens_at_fixed = [float(interp_func(fp)) for fp in valid_fp_points]

            # 构建固定点列表，包括起点和终点
            full_fp_points = [min_fp] + valid_fp_points + [max_fp]
            full_sens_points = [0.0] + sens_at_fixed + [sorted_sens[-1]]

            # 使用梯形积分法计算AUC
            auc = np.trapz(full_sens_points, full_fp_points)

    return auc


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
    if diffs:
        closest_idx = diffs.index(min(diffs))
        
        sensitivity_at_target_fp = sensitivities[closest_idx]
        threshold_at_target_fp = thresholds[closest_idx] if closest_idx < len(thresholds) else 0
        
        return sensitivity_at_target_fp, threshold_at_target_fp
    else:
        return 0, 0


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


def process_fold_predictions(fold=0, save_path="/workspace/results_froc", iou_threshold=0.01, name=None, fixed_fp_points=None):
    """
    处理指定折的预测结果，使用3D边界框计算FROC曲线
    
    Args:
        fold: 折数
        save_path: 结果保存路径
    """
    # 读取预测结果
    pred_csv_path = f"/workspace/predictions_by_fold/{name}/predictions_fold_{fold}.csv"
    if not os.path.exists(pred_csv_path):
        print(f"预测结果文件不存在: {pred_csv_path}")
        return
    
    pred_df = pd.read_csv(pred_csv_path)
    print(f"读取到 {len(pred_df)} 条预测结果")
    
    # 按患者和路径分组处理
    grouped = pred_df.groupby(['patient_key', 'path'])
    print(f"共有 {len(grouped)} 个数据样本")
    
    all_predictions_with_details = []  # 包含详细信息的预测列表
    all_ground_truths_3d = []  # 存储所有3D ground truth
    unique_patients = set()  # 用于统计病例数
    processed_masks = set()  # 避免重复处理相同的掩码
    
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
        
        print(f"标签掩码路径: {mask_path}")
        
        if mask_path is None or not mask_path.exists():
            print(f"掩码文件不存在: {mask_path}")
            continue

        # 只有当掩码未被处理过时才处理它
        if str(mask_path) not in processed_masks:
            # 加载掩码
            mask_array, mask_info = load_mask(str(mask_path))
            
            # 将掩码转换为3D边界框
            gt_boxes_3d = mask_to_3d_boxes(mask_array)
            
            # 收集3D ground truth
            all_ground_truths_3d.extend(gt_boxes_3d)
            processed_masks.add(str(mask_path))
        
        # 收集预测结果（带详细信息）
        for idx, row in group.iterrows():
            score = row['userAnnotComment.annotation']
            
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
    
    # 计算FROC曲线（包括FP详情）
    fps, sensitivities, thresholds, all_fp_details = calculate_froc_3d_with_fp_details(
        all_predictions_with_details, all_ground_truths_3d, patient_count, iou_threshold
    )
    
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
    
    # 根据阈值筛选FP点
    # 我们需要找到在阈值为target_threshold时的FP点
    filtered_fp_details = []
    for pred in all_predictions_with_details:
        score, pred_box, original_row = pred
        if score >= target_threshold:  # 高于阈值的预测会被考虑
            # 检查这个预测是否为FP（需要与GT比较）
            is_fp = True
            for gt_box in all_ground_truths_3d:
                iou = calculate_3d_iou(pred_box, gt_box)
                if iou >= iou_threshold:  # 使用与FROC计算相同的IoU阈值
                    is_fp = False
                    break
            if is_fp:
                # 将原始行数据转换为字典
                fp_row = original_row.to_dict() if hasattr(original_row, 'to_dict') else dict(original_row)

                # 添加像素坐标系下的3D边界框信息
                # pred_box格式: [z_min, y_min, x_min, z_max, y_max, x_max]
                z_min, y_min, x_min, z_max, y_max, x_max = pred_box

                # 最小角点像素坐标
                fp_row['pixel_x_min'] = x_min
                fp_row['pixel_y_min'] = y_min
                fp_row['pixel_z_min'] = z_min

                # 最大角点像素坐标
                fp_row['pixel_x_max'] = x_max
                fp_row['pixel_y_max'] = y_max
                fp_row['pixel_z_max'] = z_max

                # 中心点像素坐标
                fp_row['pixel_center_x'] = (x_min + x_max) / 2
                fp_row['pixel_center_y'] = (y_min + y_max) / 2
                fp_row['pixel_center_z'] = (z_min + z_max) / 2

                # 长宽高（像素坐标系下）
                fp_row['pixel_width'] = x_max - x_min
                fp_row['pixel_height'] = y_max - y_min
                fp_row['pixel_depth'] = z_max - z_min

                filtered_fp_details.append(fp_row)
    
    # 保存FP点信息到新的CSV文件
    if filtered_fp_details:
        fp_df = pd.DataFrame(filtered_fp_details)
        fp_output_path = f"{save_path}/fp_points_fold_{fold}_threshold_{target_threshold:.4f}.csv"
        fp_df.to_csv(fp_output_path, index=False)
        print(f"FP点信息已保存到 {fp_output_path}，共 {len(filtered_fp_details)} 个FP点")
    else:
        print("没有找到符合条件的FP点")
    
    # 打印结果
    print("FROC曲线计算完成")
    print(f"最大灵敏度: {max(sensitivities) if sensitivities else 0:.4f}")
    print(f"平均每个病例对应的最大假阳性数: {fps[sensitivities.index(max(sensitivities))] if sensitivities else 0:.4f}")

    # 计算FROC曲线下面积
    froc_auc_full = calculate_froc_auc(fps, sensitivities, fixed_fp_points=None)
    print(f"FROC曲线下面积 (完整范围 FROC-AUC): {froc_auc_full:.4f}")

    # 如果指定了固定FP点，计算限定范围内的AUC
    if fixed_fp_points is not None and len(fixed_fp_points) > 0:
        froc_auc_fixed = calculate_froc_auc(fps, sensitivities, fixed_fp_points=fixed_fp_points)
        print(f"固定FP点 {fixed_fp_points} 范围内的 FROC-AUC: {froc_auc_fixed:.4f}")

    print(f"平均每个病例出现{target_fp_per_patient}个FP时的灵敏度: {target_sensitivity:.4f}")
    print(f"平均每个病例出现{target_fp_per_patient}个FP时的置信度阈值: {target_threshold:.4f}")


if __name__ == "__main__":
    # 处理第0折的预测结果
    name = "249_neg_0"
    save_path = f"/workspace/results_froc/{name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 固定FP点数，用于计算限定范围内的FROC-AUC
    # 常见设置为 [1, 2, 4, 8, 16] 或 [1, 5, 10, 20] 等
    fixed_fp_points = [100]

    process_fold_predictions(fold=4, save_path=save_path, iou_threshold=0.01,
                            name=name, fixed_fp_points=fixed_fp_points)
