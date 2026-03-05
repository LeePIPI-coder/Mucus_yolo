"""
将 DICOM 序列或 NIfTI(.nii, .nii.gz) 文件切成 128x128 补丁，记录每个补丁在原图的层号和坐标，将补丁送入 YOLO 模型做预测，再把预测框还原回原图并展示整张带预测框的图像。
用法示例：
  python patch_predict.py --input /path/to/series_or_nii --model /path/to/best.pt --out /path/to/outdir

依赖（请在容器里安装）：pydicom, nibabel, numpy, opencv-python, pandas, ultralytics
"""
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import pydicom
import nibabel as nib
from ultralytics import YOLO

from tqdm import tqdm



def load_dicom_series(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file()]
    dicoms = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p))
            dicoms.append((p, ds))
        except Exception:
            continue
    if len(dicoms) == 0:
        raise RuntimeError('No DICOM files found in folder')
    # sort by InstanceNumber if available, else by filename
    def key_fn(item):
        ds = item[1]
        return getattr(ds, 'InstanceNumber', None) or getattr(ds, 'SliceLocation', None) or str(item[0])
    dicoms.sort(key=key_fn)
    slices = [ds.pixel_array for _, ds in dicoms]
    vol = np.stack(slices, axis=0)  # shape (Z, H, W)
    return vol


def load_nifti(path: Path):
    img = nib.load(str(path)) # 原始维度 (512, 512, 547)
    data = img.get_fdata()
    data = data.transpose(1, 0, 2) # 变换后 (512, 512, 547)
    # nibabel shape is (X, Y, Z) or (H, W, D). We'll transpose to (Z, H, W)
    if data.ndim == 3:
        # if shape is (H, W, Z) => move last to first
        if data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
            vol = np.transpose(data, (2, 0, 1))
        else:
            vol = np.transpose(data, (2, 0, 1))
    else:
        raise RuntimeError('Unsupported nifti dims: ' + str(data.shape))
    return vol


def normalize_to_uint8(img):
    # img: 2D numpy float or int
    mn = np.nanmin(img)
    mx = np.nanmax(img)
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - mn) / (mx - mn)
    arr = (scaled * 255.0).astype(np.uint8)
    return arr


def make_patches_from_volume(vol, out_images_dir: Path, mapping_csv: Path, patch_size=128, stride=None):
    # vol shape: (Z, H, W)
    Z, H, W = vol.shape
    if stride is None:
        stride = patch_size
    records = []
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # 新增：保存原始整张 slice 的目录
    slices_dir = out_images_dir.parent / 'slices'
    slices_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for z in range(Z):
        
        slice_img = vol[z]
        img_u8 = normalize_to_uint8(slice_img)
        # ensure 3 channels for YOLO (BGR)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        # img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 保存整张 slice 图作为背景（一次）
        slice_fname = f"slice_z{z:04d}.png"
        if not (slices_dir / slice_fname).exists():
            cv2.imwrite(str(slices_dir / slice_fname), img_bgr)

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
                fpath = out_images_dir / fname
                if z == 319 :
                    cv2.imwrite(str(fpath), patch)
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
                idx += 1
    if not mapping_csv.exists():
        df = pd.DataFrame.from_records(records)
        df.to_csv(mapping_csv, index=False)
        return df
    else:
        return None


def run_yolo_and_restore(model_path: Path, images_dir: Path, mapping_csv: Path, out_dir: Path, conf=0.1, device=0, batch_size=32):
    if YOLO is None:
        raise RuntimeError('ultralytics YOLO not available (install ultralytics)')
    model = YOLO(str(model_path))
    df = pd.read_csv(mapping_csv)
    image_paths = [str(images_dir / fn) for fn in df['filename'].tolist()]

    # slices 目录位置（与 make_patches_from_volume 保持一致）
    slices_dir = images_dir.parent / 'slices'

    # run inference in batches to avoid loading all images at once
    slice_canvases = {}
    slice_counts = {}
    Predict_slice_exist = []

    # 进度条初始化
    pb = tqdm(total=len(os.listdir(slices_dir)), desc='Detecting', unit='slice') if tqdm is not None else None
    simple_count = 0

    for offset in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[offset: offset + batch_size]
        results = model(batch_paths, device=device, conf=conf, verbose=False)
        for j, res in enumerate(results):
            i = offset + j
            row = df.iloc[i]
            z = int(row['slice'])
            x_off = int(row['x'])
            y_off = int(row['y'])
            orig_w = int(row['orig_w'])
            orig_h = int(row['orig_h'])
            # prepare canvas if not exists: 用原始 slice 作为背景
            if z not in slice_canvases:
                bg_path = slices_dir / f"slice_z{z:04d}.png"
                if bg_path.exists():
                    bg = cv2.imread(str(bg_path))
                    if bg is None:
                        slice_canvases[z] = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    else:
                        # 若尺寸不匹配，调整
                        if (bg.shape[1], bg.shape[0]) != (orig_w, orig_h):
                            bg = cv2.resize(bg, (orig_w, orig_h))
                        # bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        slice_canvases[z] = bg.copy()
                else:
                    slice_canvases[z] = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                slice_counts[z] = 0
            # if there are no boxes, skip
            boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else np.array([])
            scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else None
            classes = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, 'cls') else None
            for bi, box in enumerate(boxes):
                if z not in Predict_slice_exist:
                    Predict_slice_exist.append(z)
                x1, y1, x2, y2 = box
                # map back to original coordinates
                ox1 = int(x1) + x_off
                oy1 = int(y1) + y_off
                ox2 = int(x2) + x_off
                oy2 = int(y2) + y_off
                color = (0, 255, 0)
                cv2.rectangle(slice_canvases[z], (ox1, oy1), (ox2, oy2), color, 2)
                label = ''
                if scores is not None and classes is not None:
                    # label = f"{int(classes[bi])}:{scores[bi]:.2f}"
                    label = f""
                    cv2.putText(slice_canvases[z], label, (ox1, max(oy1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    # cv2.putText(slice_canvases[z], label, (ox1, max(oy1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 更新进度
            
        pb.update(1)



    pb.close()
    # save per-slice visualizations
    out_dir.mkdir(parents=True, exist_ok=True)
    for z, canvas in slice_canvases.items():
        out_path = out_dir / f"pred_slice_{z:04d}.png"
        cv2.imwrite(str(out_path), canvas)
    # return [str(out_dir / f"pred_slice_{z:04d}.png") for z in slice_canvases.keys()]
    print('Predicted slices:', Predict_slice_exist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=r'/data/nifti_files/collect_test_image/E0001009_20091120.nii.gz',help='DICOM folder or NIfTI file (.nii or .nii.gz)')
    parser.add_argument('--model', '-m', default=r'Train_result/Mucus_249_neg_0/Train_20260302_fold0/weights/best.pt',help='YOLO model .pt')
    parser.add_argument('--out', '-o', default=r'/data/Predict',help='output directory')
    parser.add_argument('--patch', type=int, default=128, help='patch size (default 128)')
    parser.add_argument('--stride', type=int, default=64, help='stride, default = patch size')
    parser.add_argument('--conf', type=float, default=0.1, help='YOLO confidence threshold')
    parser.add_argument('--device', type=int, default=0, help='device id for inference')
    parser.add_argument('--batch-size', type=int, default=64, help='inference batch size')
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    images_dir = out / 'images'
    mapping_csv = out / 'mapping.csv'
    preds_dir = out / 'preds'

    # load volume
    if inp.is_dir():
        if pydicom is None:
            raise RuntimeError('pydicom not installed but input is a folder of DICOMs')
        vol = load_dicom_series(inp)
    elif inp.is_file():
        if inp.suffix in ('.nii', '.gz'):
            if nib is None:
                raise RuntimeError('nibabel not installed but input is a nifti')
            vol = load_nifti(inp)
        else:
            raise RuntimeError('Unsupported input file type: ' + str(inp))
    else:
        raise RuntimeError('Input path not found')

    df = make_patches_from_volume(vol, images_dir, mapping_csv, patch_size=args.patch, stride=args.stride)
    pred_files = run_yolo_and_restore(Path(args.model), images_dir, mapping_csv, preds_dir, conf=args.conf, device=args.device, batch_size=args.batch_size)
    print('Saved prediction images:', pred_files)


if __name__ == '__main__':
    main()
