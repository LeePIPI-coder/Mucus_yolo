from ultralytics import YOLO

# Load a model
model = YOLO("/workspace/Train_result/Mucus_249_neg_0.1/Train_fold4/weights/best.pt")

# Customize validation settings
metrics = model.val(data="/data/yolo_dataset_249/Kfold_neg_0.1/fold4.yaml", imgsz=512, batch=256, device="0")

# Calculate accuracy from confusion matrix
# Ultralytics confusion matrix layout for binary classification:
#              Predicted
#              Neg  Pos
# Actual Neg [TN  FP]
# Actual Pos [FN  TP]


# if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
#     cm = metrics.confusion_matrix.matrix
#     # cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
#     tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
#     acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
#     print(f"\n=== Additional Metrics ===")
#     print(f"Precision (Pre): {metrics.box.mp:.4f}")
#     print(f"Recall: {metrics.box.mr:.4f}")
#     print(f"Accuracy (Acc): {acc:.4f}")
#     # F1 = 2 * Pre * Recall / (Pre + Recall)
#     f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr) if (metrics.box.mp + metrics.box.mr) > 0 else 0.0
#     print(f"F1 Score: {f1:.4f}")
#     print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
# else:
#     print(f"\n=== Metrics ===")
#     print(f"Precision (Pre): {metrics.box.mp:.4f}")
#     print(f"Recall: {metrics.box.mr:.4f}")
#     f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr) if (metrics.box.mp + metrics.box.mr) > 0 else 0.0
#     print(f"F1 Score: {f1:.4f}")
#     print("Confusion matrix not available for accuracy calculation")
