import cv2
import os
import numpy as np
import argparse
import tqdm
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from utils.utils import load_dicom_series, pre_processing, post_processing, coord_vox2pat
import ast
from test import predict


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
def process_fold_predictions(csv_path, base_model_path_pattern, output_dir="./predictions_by_fold"):
    """
    Load the model weights for each fold based on CSV fold info,
    run predictions, and save results to separate CSV files per fold.

    Args:
        csv_path: Path to CSV with 'val_fold' and 'dicom_path' columns
        base_model_path_pattern: Model path pattern, e.g. "Train_result/Mucus_249_neg_0/Train_20260302_fold{fold}/weights/best.pt"
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    folds = df['val_fold'].unique()

    print(f"Found {len(folds)} distinct folds: {folds}")

    for fold in folds:
        if fold > 0:
            continue
        print(f"\nProcessing fold {fold}...")

        fold_data = df[df['val_fold'] == fold]
        print(f"Fold {fold} has {len(fold_data)} samples")

        # model_path = base_model_path_pattern.format(fold=fold)
        model_path = "/workspace/Train_result/Mucus_249_neg_0/Train_fold1/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"Warning: model file not found: {model_path}")
            print(f"Skipping fold {fold}")
            continue

        print(f"Using model: {model_path}")

        model = YOLO(model_path)

        all_fold_results = []

        for idx, row in fold_data.iterrows():
            # if idx < 118:
            #     continue
            dicom_path = row['dicom_path']

            # patient_key = row['patient_key']
            # if patient_key != "E0001009_20091120":
            #     continue
            if not os.path.exists(dicom_path):
                print(f"Warning: DICOM path not found: {dicom_path}")
                continue

            print(f"Predicting: {dicom_path}")

            try:
                dicom_dir = find_depth_dir(dicom_path)
                result_df = predict(dicom_dir, model)

                result_df['patient_key'] = row['patient_key']
                result_df['val_fold'] = fold

                all_fold_results.append(result_df)

            except Exception as e:
                print(f"Error predicting {dicom_path}: {str(e)}")
                continue

        if all_fold_results:
            combined_results = pd.concat(all_fold_results, ignore_index=True)

            output_file = os.path.join(output_dir, f"Predictions_fold_{fold}.csv")
            combined_results.to_csv(output_file, index=False)
            print(f"Fold {fold} predictions saved to: {output_file}")
            print(f"Total {len(combined_results)} predictions")
        else:
            print(f"Fold {fold} produced no predictions")

    print("\nAll folds complete!")


def main():
    parser = argparse.ArgumentParser(description="Load model weights per fold and run predictions")
    parser.add_argument("--csv_path", type=str,
                        default="/workspace/test_input_csv/249_neg_0/test_fold_data.csv",
                        help="CSV file path with val_fold and dicom_path columns")
    parser.add_argument("--base_model_path", type=str,
                        default="Train_result/Mucus_249_neg_0/Train_fold{fold}/weights/best.pt",
                        help="Model path pattern, {fold} will be replaced with the fold number")
    parser.add_argument("--output_dir", type=str,
                        default="./all_results/predictions_by_fold/249_test_me",
                        help="Output directory")

    args = parser.parse_args()

    print(f"CSV path: {args.csv_path}")
    print(f"Base model path: {args.base_model_path}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return

    process_fold_predictions(args.csv_path, args.base_model_path, args.output_dir)


if __name__ == "__main__":
    main()