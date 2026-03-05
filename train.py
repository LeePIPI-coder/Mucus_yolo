from ultralytics import YOLO

if __name__ == "__main__":
# ---------进行交叉验证时，微调更改data、project、name参数-----------
    for i in range(5):
        if i == 0:
            continue
        print(f"正在训练第{i}折数据")
        model = YOLO("/home/LJR/Mucus_project/demo_mucusAlgorithms/yolo11n.pt")
        train_results = model.train(
            data=f"dataset/yolo_dataset_241/Kfold_neg_0.1/fold{i}.yaml",  # Path to dataset configuration file
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
            # cache='ram', # 是否启用内存中缓存数据集图像
            project="Train_result/Mucus_neg_0.1",
            name=f"Train_20260303_fold{i}",
            # resume=Trues
        )