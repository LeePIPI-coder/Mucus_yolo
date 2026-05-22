from ultralytics import YOLO
import os 
import cv2
import random


def display_test_out_png(model_weight:str, label_dir:str, save_dir:str):
    """
    function:保存在测试集上的检测结果可视化图片以及其对应的标签图片用于对比实验

    Args:
        model_weight (str): 模型权重路径
        label_dir (str): 标签目录路径
        save_dir (str): 保存目录路径

    """
    label_lists = os.listdir(label_dir)
    # 选取的文件序列号
    pre_range = slice(0,100)
    Pre_label_lists = label_lists[pre_range]
    
    # 随机抽取
    # num_to_sample = 20
    # Pre_label_lists = random.sample(label_lists, num_to_sample)
    
    label_paths = [os.path.join(label_dir, x) for x in Pre_label_lists]
    image_paths = [path.replace('labels', 'images').replace('txt', 'png') for path in label_paths]

    # 创建保存路径
    os.makedirs(f"/home/LJR/Mucus_project/demo_mucusAlgorithms/Output_display/{save_dir}", exist_ok=True)
    
    # Load a model
    model = YOLO(model_weight)  # pretrained YOLO26n model
    # Run batched inference on a list of images
    results = model(image_paths[pre_range], device=0, conf=0.1)  # return a list of Results objects
    label_detect_list = label_paths
    # Process results list
    idex_list = []
    num = 0
    for idex, result in enumerate(results):
        # if result.boxes:
        original_img = cv2.imread(result.path)
        if len(result.boxes.cls) != 0:
            num += 1      
        img_pre = original_img.copy()    # 用于绘制预测框
        img_label = original_img.copy()  # 用于绘制标签框
        
        # 2. 绘制预测框 (Prediction)
        boxes = result.boxes.xyxy.cpu().numpy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img_pre, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2)
        
        # 3. 绘制标签框 (Label)
        label_path = label_detect_list[idex]
        h, w = img_label.shape[:2]
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines: # 遍历所有行，而不仅仅是第一行
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, bw, bh = map(float, parts)
                lx1 = int((xc - bw / 2) * w)
                ly1 = int((yc - bh / 2) * h)
                lx2 = int((xc + bw / 2) * w)
                ly2 = int((yc + bh / 2) * h)
                
                        # 防止越界
                x1 = max(0, lx1)
                y1 = max(0, ly1)
                x2 = min(w - 1, lx2)
                y2 = min(h - 1, ly2)
                cv2.rectangle(img_label, (lx1, ly1), (lx2, ly2), (0, 0, 255), thickness=2) # 改为红色区分

        # 4. 左右拼接图片
        # 使用 cv2.hconcat 将 img_pre(左) 和 img_label(右) 拼接
        combined_img = cv2.hconcat([img_pre, img_label])
        
        # 5. 保存拼接后的图片
        png_name = result.path.split('/')[-1]
        save_path = f"/home/LJR/Mucus_project/demo_mucusAlgorithms/Output_display/{save_dir}/{png_name}"
        cv2.imwrite(save_path, combined_img)
    print(f"共检出{num}张")
        # print(f"Saved combined image to: {save_path}")

if __name__ == "__main__":
    model_weight = r"/home/LJR/Mucus_project/demo_mucusAlgorithms/Mucus_neg_0/Train_20260227_fold2/weights/best.pt"
    label_dir = r"/home/LJR/Mucus_project/demo_mucusAlgorithms/dataset/yolo_dataset/test/labels"
    save_dir = '20260302_neg_0_Ktest' 
    display_test_out_png(model_weight, label_dir, save_dir)
