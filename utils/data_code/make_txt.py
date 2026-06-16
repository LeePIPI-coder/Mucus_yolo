import os
import cv2
import numpy as np
import cc3d
from PIL import Image
from tqdm import tqdm
import argparse

def create_yolo_annotations_with_cc3d(image_dir, mask_dir, output_txt_dir, category_id=0):
    """
    Generate YOLO-format bounding box annotation files.

    Uses connected component analysis to identify individual mucus regions
    in medical images.

    Args:
        image_dir (str): Path to image file directory
        mask_dir (str): Path to mask file directory
        output_txt_dir (str): Output directory for YOLO annotation files
        category_id (int): Target class ID (default 0 = mucus class)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_txt_dir, exist_ok=True)

    # Get all PNG or NPY files in the image directory, sorted by filename
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".npy")])

    # Iterate over all image files with a progress bar
    for img_file in tqdm(image_files):
        # Build full file paths
        img_path = os.path.join(image_dir, img_file)
        # Corresponding mask file path (strip extension, add .png)
        mask_path = os.path.join(mask_dir, os.path.splitext(img_file)[0] + ".png")
        # Corresponding YOLO annotation file path (.txt format)
        txt_output_path = os.path.join(output_txt_dir, os.path.splitext(img_file)[0] + ".txt")

        # Skip this image if the corresponding mask file doesn't exist
        if not os.path.exists(mask_path):
            continue

        # (commented out: originally read image dimensions from the image file)
        # array_img = np.load(img_path)
        # width, height, channel = array_img.shape
        # print(width, height, channel)

        # Load binary mask image (grayscale mode)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Get mask dimensions
        width, height = mask.shape
        # print(width, height)

        # Binarize the mask (pixels > 0 become 1, others become 0)
        binary_mask = (mask > 0).astype(np.uint8)

        # Connected component labeling: identify independent connected regions
        # connectivity=8 uses 8-neighbor connectivity (includes diagonals)
        # labels_out: each element is a region label, starting from 1; 0 = background
        labels_out = cc3d.connected_components(binary_mask, connectivity=8)
        # Number of detected objects = the maximum label value
        num_objects = labels_out.max()
        # print(mask_path, ": ", num_objects)

        # Store YOLO-format annotation lines
        yolo_lines = []

        # Iterate over each detected object (labels start from 1; 0 is background)
        for obj_label in range(1, num_objects + 1):
            # Extract the mask region of the current object
            obj_mask = (labels_out == obj_label).astype(np.uint8)

            # Set bounding box margin in pixels
            margin = 5
            # Find all non-zero pixel coordinates of the object mask
            coords = cv2.findNonZero(obj_mask)
            # Compute the minimum bounding rectangle containing all non-zero pixels
            x, y, w, h = cv2.boundingRect(coords)

            # Adjust coordinates: apply margin while keeping within image bounds
            # Adjust x: extend left by margin, but not below 0
            x = max(x - margin, 0)
            # Adjust y: extend up by margin, but not below 0
            y = max(y - margin, 0)
            # Adjust width: extend right by margin, but not beyond image width
            w = min(w + 2 * margin, width - x)
            # Adjust height: extend down by margin, but not beyond image height
            h = min(h + 2 * margin, height - y)

            # Convert to YOLO format (normalized coordinates)
            # Bounding box center x coordinate (normalized to 0-1)
            x_center = (x + w / 2) / width
            # Bounding box center y coordinate (normalized to 0-1)
            y_center = (y + h / 2) / height
            # Bounding box width (normalized to 0-1)
            w_norm = w / width
            # Bounding box height (normalized to 0-1)
            h_norm = h / height

            # Generate YOLO-format annotation line: class_id x_center y_center width height
            yolo_lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            # print(yolo_lines[-1])  # Print the last generated line for debugging

        # Save YOLO annotation file
        with open(txt_output_path, "w") as f:
            for line in yolo_lines:
                f.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make txt")
    parser.add_argument("-path", default=r"/data/yolo_dataset_249", help="dataset root path")
    parser.add_argument("-set", default="All" ,help="make yolo bbox txt of the set type")
    args = parser.parse_args()

    create_yolo_annotations_with_cc3d(
        image_dir="{}/{}/images".format(args.path, args.set),
        mask_dir="{}/{}/masks".format(args.path, args.set),
        output_txt_dir="{}/{}/labels".format(args.path, args.set)
    )
