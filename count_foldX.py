from pathlib import Path
import random

root = Path("/home/LJR/Mucus_project/demo_mucusAlgorithms/dataset/yolo_dataset_241/train/images")
paths = list(root.iterdir())
print("总数:", len(paths))
for p in random.sample(paths, 10):
    print(p.name)