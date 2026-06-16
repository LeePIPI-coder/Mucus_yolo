from ultralytics import YOLO

if __name__ == "__main__":
# --------- For cross-validation, adjust data/project/name parameters -----------
    for i in range(5):
        print(f"Training fold {i}")
        model = YOLO("/workspace/yolo11n.pt")
        train_results = model.train(
            data=f"/data/yolo_dataset_249/Kfold_neg_0.1/fold{i}.yaml",  # Path to dataset configuration file
            epochs=100,  # Number of training epochs
            batch=256,
            imgsz=512,  # Image size for training
            scale=0.2,  # 
            degrees=5.0,
            fliplr=0.5,
            flipud=0.5,
            copy_paste=0.0,
            mixup=0.0,
            mosaic=0.0,
            amp=True,
            save_period=10,
            workers=4,
            device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
            # cache='ram',  # Enable in-memory caching of dataset images
            project="Train_result/Mucus_249_neg_0.1",
            name=f"Train_fold{i}",
            # resume=Trues
        )