import os
import cv2
import numpy as np
import cc3d
from PIL import Image
from tqdm import tqdm
import argparse

def create_yolo_annotations_with_cc3d(image_dir, mask_dir, output_txt_dir, category_id=0):
    """
    为YOLO目标检测模型生成边界框标注文件
    使用连通组件分析技术识别医学影像中的多个粘液区域
    
    Args:
        image_dir (str): 图像文件目录路径
        mask_dir (str): 掩码文件目录路径
        output_txt_dir (str): YOLO标注文件输出目录
        category_id (int): 目标类别ID（默认为0，表示粘液类别）
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_txt_dir, exist_ok=True)
    
    # 获取图像目录中所有PNG或NPY格式的文件，并按文件名排序
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".npy")])

    # 使用进度条遍历所有图像文件
    for img_file in tqdm(image_files):
        # 构建完整的文件路径
        img_path = os.path.join(image_dir, img_file)
        # 构建对应的掩码文件路径（去掉扩展名后加.png）
        mask_path = os.path.join(mask_dir, os.path.splitext(img_file)[0] + ".png")
        # 构建对应的YOLO标注文件路径（.txt格式）
        txt_output_path = os.path.join(output_txt_dir, os.path.splitext(img_file)[0] + ".txt")

        # 如果对应的掩码文件不存在，跳过此图像
        if not os.path.exists(mask_path):
            continue

        # 注释掉的代码：原本用于从图像文件获取尺寸，现在改为从掩码获取
        # array_img = np.load(img_path)
        # width, height, channel = array_img.shape
        # print(width, height, channel)

        # 加载二值掩码图像（灰度模式）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 获取掩码图像的宽度和高度
        width, height = mask.shape
        # print(width, height)

        # 将掩码转换为二值图像（大于0的像素设为1，其余为0）
        binary_mask = (mask > 0).astype(np.uint8)

        # 连通组件标记：识别图像中独立的连通区域
        # connectivity=8表示使用8邻域连通性（包括对角线方向）
        # label_out: 3D 数组，每个元素表示一个连通区域，值为连通区域编号，从1开始，0表示背景。
        labels_out = cc3d.connected_components(binary_mask, connectivity=8)
        # 获取最大的标签值，即检测到的目标数量
        num_objects = labels_out.max()
        # print(mask_path, ": ", num_objects)
        
        # 存储YOLO格式的标注行
        yolo_lines = []

        # 遍历每个检测到的目标（标签从1开始，0是背景）
        for obj_label in range(1, num_objects + 1):
            # 提取当前目标的掩码区域
            obj_mask = (labels_out == obj_label).astype(np.uint8)

            # 设置边界框边距（像素）
            margin = 5
            # 找到目标掩码中所有非零像素的坐标
            coords = cv2.findNonZero(obj_mask)
            # 计算包含所有非零像素的最小外接矩形
            x, y, w, h = cv2.boundingRect(coords)

            # 坐标调整：应用边距并确保不超出图像边界
            # 调整x坐标：向左扩展边距，但不少于0
            x = max(x - margin, 0)
            # 调整y坐标：向上扩展边距，但不少于0
            y = max(y - margin, 0)
            # 调整宽度：向右扩展边距，但不超出图像宽度
            w = min(w + 2 * margin, width - x)
            # 调整高度：向下扩展边距，但不超出图像高度
            h = min(h + 2 * margin, height - y)

            # 转换为YOLO格式（归一化坐标）
            # 计算边界框中心点的x坐标（归一化到0-1）
            x_center = (x + w / 2) / width
            # 计算边界框中心点的y坐标（归一化到0-1）
            y_center = (y + h / 2) / height
            # 计算边界框宽度（归一化到0-1）
            w_norm = w / width
            # 计算边界框高度（归一化到0-1）
            h_norm = h / height

            # 生成YOLO格式的标注行：类别ID 中心x 中心y 宽度 高度
            yolo_lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            # 打印当前生成的标注行（用于调试）
            # print(yolo_lines[-1])
        
        # 保存YOLO标注文件
        with open(txt_output_path, "w") as f:
            # 将每个标注行写入文件，每行一个目标
            for line in yolo_lines:
                f.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make txt")
    parser.add_argument("-path", default=r"/home/LJR/Mucus_project/demo_mucusAlgorithms/dataset/yolo_dataset_249", help="dataset root path")
    parser.add_argument("-set", default="valid" ,help="make yolo bbox txt of the set type")
    args = parser.parse_args()


    # 사용 예시 
    create_yolo_annotations_with_cc3d(
        image_dir="{}/{}/images".format(args.path, args.set),     
        mask_dir="{}/{}/masks".format(args.path, args.set),      
        output_txt_dir="{}/{}/labels".format(args.path, args.set) 
    )