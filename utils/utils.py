import SimpleITK as sitk
import numpy as np
import pandas as pd

def read_image(image_path):
    image = sitk.ReadImage(image_path)
    origin  = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    StudyInstanceUID = image.GetMetaData('0020|000d') if image.HasMetaDataKey('0020|000d') else None
    SeriesInstanceUID = image.GetMetaData('0020|000e') if image.HasMetaDataKey('0020|000e') else None
    
    return {
        'origin': origin,
        'spacing': spacing,
        'direction': direction,
        'StudyInstanceUID': StudyInstanceUID,
        'SeriesInstanceUID': SeriesInstanceUID
    }

def load_dicom_series(dicom_dir):
    """
    주어진 디렉터리에서 DICOM 시리즈를 읽어 SimpleITK 이미지 객체로 반환합니다.
    
    Parameters:
        dicom_dir (str): DICOM 시리즈가 저장된 디렉터리 경로

    Returns:
        image (SimpleITK.Image): 3D DICOM 이미지 객체
        spacing (tuple): (spacing_x, spacing_y, spacing_z)
        size (tuple): (size_x, size_y, size_z)
    """
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)

    if not series_IDs:
        raise FileNotFoundError(f"No DICOM series found in directory: {dicom_dir}")

    # 가장 많은 dicom file을 가진 시리즈 선택
    max_files = 0
    max_series_id = None
    for series_id in series_IDs:
        file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
        if len(file_names) > max_files:
            max_files = len(file_names)
            max_series_id = series_id
    if max_series_id is None:
        raise FileNotFoundError("No DICOM series with files found")
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, max_series_id)

    reader.SetFileNames(series_file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()
    #info = read_image(series_file_names[len(series_file_names) // 2])
    # 3D 이미지 객체에서 직접 메타데이터 추출
    info = {
        'origin': image.GetOrigin(),       # 3D 볼륨의 시작점 (정확함)
        'spacing': image.GetSpacing(),     # (pixel_x, pixel_y, slice_thickness)
        'direction': image.GetDirection(), # 3D 방향 코사인 행렬
        # UID는 메타데이터 딕셔너리에서 가져와야 함 (SimpleITK 방식)
        'StudyInstanceUID': reader.GetMetaData(0, '0020|000d') if reader.HasMetaDataKey(0, '0020|000d') else None,
        'SeriesInstanceUID': reader.GetMetaData(0, '0020|000e') if reader.HasMetaDataKey(0, '0020|000e') else None
    }

    return sitk.GetArrayFromImage(image), info

def coord_vox2pat(vox, origin, spacing, direction):
    origin = np.array(origin)
    spacing = np.array(spacing)
    direction = np.array(direction)
    direction_matrix = direction.reshape(3, 3)
    transformation_matrix = direction_matrix * spacing
    vox = np.array(vox)
    patient_coord = transformation_matrix.dot(vox) + origin
    return tuple(patient_coord)

def coord_pat2vox(pat, origin, spacing, direction):
    origin = np.array(origin)
    spacing = np.array(spacing)
    direction = np.array(direction)
    direction_matrix = direction.reshape(3, 3)
    transformation_matrix = direction_matrix * spacing
    pat = np.array(pat)
    voxel_coord = np.linalg.inv(transformation_matrix).dot(pat - origin)
    return tuple(voxel_coord)

def iou_3d(box1, box2):
    x1_min, y1_min, z1_min = box1[0] - box1[3]/2, box1[1] - box1[4]/2, box1[2] - box1[5]/2
    x1_max, y1_max, z1_max = box1[0] + box1[3]/2, box1[1] + box1[4]/2, box1[2] + box1[5]/2
    x2_min, y2_min, z2_min = box2[0] - box2[3]/2, box2[1] - box2[4]/2, box2[2] - box2[5]/2
    x2_max, y2_max, z2_max = box2[0] + box2[3]/2, box2[1] + box2[4]/2, box2[2] + box2[5]/2
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    zi_min = max(z1_min, z2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    zi_max = min(z1_max, z2_max)
    inter_dx = max(0, xi_max - xi_min)
    inter_dy = max(0, yi_max - yi_min)
    inter_dz = max(0, zi_max - zi_min)
    inter_vol = inter_dx * inter_dy * inter_dz
    vol1 = box1[3]*box1[4]*box1[5]
    vol2 = box2[3]*box2[4]*box2[5]
    union = vol1 + vol2 - inter_vol
    return inter_vol / union if union > 0 else 0

def nms_3d(boxes, scores, iou_threshold=0.001):
    """
    Perform 3D NMS (matching new schema: x, y, z, diax, diay, diaz).
    Args:
      boxes: (N, 6) numpy array (center_x, center_y, center_z, diax, diay, diaz)
      scores: (N,) confidence values
      iou_threshold: float
    Returns:
      indices of boxes to keep
    """
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = np.array([iou_3d(boxes[i], boxes[j]) for j in idxs[1:]])
        idxs = idxs[1:][ious <= iou_threshold]
    return keep

def pre_processing(image):
    image = np.array(image)
    image = np.clip(image, -1000, 400)
    image = (image - (-1000)) / (400 - (-1000))
    image = (image * 255).astype(np.uint8)
    return image

def post_processing(df, iou_threshold=0.001):
    # Group by SeriesInstanceUID
    col_candidates = [c for c in df.columns if c.lower() == 'seriesinstanceuid']
    if not col_candidates:
        raise RuntimeError("SeriesInstanceUID 컬럼을 찾지 못했습니다!")  # 왜 not found 라고 나오지?
    series_uid_col = col_candidates[0]

    results = []
    for seriesuid, sub_df in df.groupby(series_uid_col):
        # "userAnnotComment.annotation" 기준 내림차순 정렬
        if 'detector_score' not in sub_df.columns:
            raise RuntimeError("No detector_score column found!")

        sub_sorted = sub_df.sort_values('detector_score', ascending=False).reset_index(drop=True)

        boxes = np.array([
            [
                sub_sorted.iloc[i]['x'],
                sub_sorted.iloc[i]['y'],
                sub_sorted.iloc[i]['z'],
                sub_sorted.iloc[i]['diameter_x'],
                sub_sorted.iloc[i]['diameter_y'],
                sub_sorted.iloc[i]['diameter_z']
            ]
            for i in range(len(sub_sorted))
        ])
        scores = sub_sorted['detector_score'].values.astype(float)
        if len(boxes) == 0:
            continue
        keep_idx = nms_3d(boxes, scores, iou_threshold)
        results.append(sub_sorted.iloc[keep_idx])

    if len(results) > 0:
        filtered_df = pd.concat(results, axis=0, ignore_index=True)
    else:
        filtered_df = pd.DataFrame(columns=df.columns)

    return filtered_df
