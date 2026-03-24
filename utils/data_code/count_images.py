import os
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gz"}


def count_images(image_dir: str) -> tuple[int, Counter]:
    """
    统计指定目录（不递归）下的图片文件数量。
    返回 (总数, 按扩展名统计 Counter)。
    """
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"目录不存在: {image_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"不是目录: {image_dir}")

    by_ext: Counter = Counter()
    total = 0

    # 用 scandir 更快，且不加载全部文件名到内存
    with os.scandir(p) as it:
        for entry in it:
            if not entry.is_file():
                continue
            ext = Path(entry.name).suffix.lower()
            # 处理image数据
            # img = Image.open(entry.path)
            # 处理nupmy数据
            # image = np.load(entry.path)
            if ext in IMAGE_EXTS:
                total += 1
                by_ext[ext] += 1

    return total, by_ext


if __name__ == "__main__":
    image_dir = "/home/LJR/Mucus_project/demo_mucusAlgorithms/dataset/nifti_files/collect_train_image"
    total, by_ext = count_images(image_dir)
    print(f"目录: {image_dir}")
    print(f"图片总数: {total}")
    if by_ext:
        print("按扩展名统计:")
        for ext, n in sorted(by_ext.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {ext}: {n}")
