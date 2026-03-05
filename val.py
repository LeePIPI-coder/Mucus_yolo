from ultralytics import YOLO

# Load a model
model = YOLO("/home/LJR/Mucus_project/demo_mucusAlgorithms/Train_result/Mucus_neg_0/Train_20260227_fold1/weights/best.pt")

# Customize validation settings
metrics = model.val(data="/home/LJR/Mucus_project/demo_mucusAlgorithms/dataset/yolo_dataset_241/Kfold_neg_0/fold1.yaml", imgsz=512, batch=256, conf=0.01, iou=0.7, device="0")