import os
import cv2
import random

def draw_yolo_boxes(
    label_path,
    images_dir,
    output_path=None,
    color=(0, 255, 0),
    thickness=2
):
    """
    根据 YOLO label 文件，在对应 png 图片上绘制检测框

    Args:
        label_path (str): labels/xxx.txt 的完整路径
        images_dir (str): images 目录路径
        output_path (str): 保存结果的路径（None 表示不保存）
        color (tuple): BGR 颜色
        thickness (int): 框线粗细

    Returns:
        img (ndarray): 画好框的图像
    """

    # 1. label -> image 文件名
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    image_path = os.path.join(images_dir, base_name + ".png")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 2. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]

    # 3. 读取 label
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id, xc, yc, bw, bh = map(float, parts)

        # 4. YOLO -> 像素坐标
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # 防止越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        # 5. 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            img,
            str(int(cls_id)),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    os.makedirs(output_path, exist_ok=True)
    # 6. 保存结果
    if output_path is not None:
        output_path = os.path.join(output_path, f'{base_name}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    label_dir = r"/home/LJR/Mucus_project/demo_mucusAlgorithms/dataset/yolo_dataset_241/train/labels"
    image_dir = r"/home/LJR/Mucus_project/demo_mucusAlgorithms/dataset/yolo_dataset_241/train/images"
    output_path = r"Output_display/labels"
    label_lists = os.listdir(label_dir)
    # ------------1单张-----------------
    # label = label_lists[100:200]
    # label_path =  os.path.join(label_dir,label)
    # draw_yolo_boxes(label_path=label_path,images_dir=image_dir,output_path=output_path)
    
    # -----------2序列-----------------
    # label_list = label_lists[:100]
    # label_list = [os.path.join(label_dir, x) for x in label_list]
    
    # ---------3随机抽取序列-----------
    # 获取所有标签文件名
    label_lists = os.listdir(label_dir)
    # --- 随机抽取核心逻辑 ---
    num_to_sample = 200  # 设置你想抽取的数量
    # 防止请求数量超过文件夹实际文件数
    sample_count = min(num_to_sample, len(label_lists)) 
    # 随机从列表中选出样本（不重复）
    random_label_list = random.sample(label_lists, sample_count)
    # 拼接完整路径
    label_full_paths = [os.path.join(label_dir, x) for x in random_label_list]
    
    for label_path in label_full_paths:
        draw_yolo_boxes(label_path=label_path,images_dir=image_dir,output_path=output_path)
    
