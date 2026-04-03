from ultralytics import YOLO

# Load a model
model = YOLO("/workspace/Train_result/Mucus_241_neg_0/Train_20260228_fold4/weights/best.pt")

# Customize validation settings
metrics = model.val(data="/data/yolo_dataset_241/Kfold_neg_0/fold4.yaml", imgsz=512, batch=256, device="0")