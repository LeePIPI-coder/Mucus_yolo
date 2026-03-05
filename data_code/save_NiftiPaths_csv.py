"""
从 nifti_files 目录下，找出名称包含 'image' 的子目录中的全部 .nii.gz 文件路径，
并保存到 CSV 文件中。
"""
import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 nifti image 目录下的 .nii.gz 路径并保存为 CSV")
    parser.add_argument(
        "--root",
        default="/data/nifti_files",
        help="nifti_files 根目录",
    )
    parser.add_argument(
        "--out",
        default="/data/nifti_image_paths.csv",
        help="输出 CSV 文件路径",
    )
    parser.add_argument(
        "--keyword",
        default="image",
        help="子目录名需包含的关键字（默认: image）",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    # 收集：名称包含 keyword 的子目录下所有 .nii.gz 的绝对路径
    paths: list[tuple[str]] = []  # (split_or_dirname, path)

    for item in root.iterdir():
        if not item.is_dir():
            continue
        if args.keyword.lower() not in item.name.lower():
            continue
        for f in item.iterdir():
            if not f.is_file():
                continue
            if f.suffix == ".gz" and f.stem.endswith(".nii"):
                full_path = str(f.resolve())
                paths.append((full_path))
                

    paths.sort(key=lambda x: (x[0]))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path"])
        for path in paths:
            writer.writerow([path])

    print(f"共写入 {len(paths)} 条路径 -> {out_path}")


if __name__ == "__main__":
    main()
