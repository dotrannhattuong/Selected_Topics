import os
import json
import glob
import cv2
import numpy as np
import skimage.io as sio
from sklearn.model_selection import train_test_split

def generate_coco_json(folders, root_dir, output_json, start_image_id=0, start_annotation_id=0):
    """
    Generate COCO-style JSON annotation from folders containing image.tif and class*.tif masks.
    """
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    image_id = start_image_id
    annotation_id = start_annotation_id
    category_ids = set()

    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        img_path = os.path.join(folder_path, "image.tif")
        img = cv2.imread(img_path)

        if img is None:
            print(f"[Warning] Cannot read image: {img_path}, skipping...")
            continue

        height, width = img.shape[:2]
        coco_output["images"].append({
            "file_name": os.path.relpath(img_path, root_dir),
            "height": height,
            "width": width,
            "id": image_id
        })

        # Process each class mask
        for mask_path in glob.glob(os.path.join(folder_path, "class*.tif")):
            try:
                category_id = int(os.path.basename(mask_path).replace("class", "").replace(".tif", ""))
            except ValueError:
                print(f"[Warning] Invalid category from filename: {mask_path}")
                continue

            category_ids.add(category_id)
            mask = sio.imread(mask_path)
            if mask is None:
                print(f"[Warning] Cannot read mask: {mask_path}, skipping...")
                continue
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # For each unique instance ID in the mask (excluding 0)
            for inst_id in np.unique(mask):
                if inst_id == 0:
                    continue
                instance_mask = (mask == inst_id).astype(np.uint8)
                if instance_mask.sum() == 0:
                    continue

                polygons = mask_to_polygons(instance_mask)
                for poly in polygons:
                    bbox = get_bbox_from_poly(poly)
                    area = cv2.contourArea(np.array(poly).reshape(-1, 2))
                    coco_output["annotations"].append({
                        "segmentation": [poly],
                        "iscrowd": 0,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "id": annotation_id
                    })
                    annotation_id += 1

        image_id += 1

    # Add category info
    for cid in sorted(category_ids):
        coco_output["categories"].append({
            "id": cid,
            "name": f"class{cid}",
            "supercategory": "none"
        })

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=2)

    return image_id, annotation_id


def mask_to_polygons(mask):
    """
    Convert binary mask to polygon(s) using OpenCV contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6:  # at least 3 points
            polygons.append(contour)
    return polygons


def get_bbox_from_poly(poly):
    """
    Get [x, y, width, height] bounding box from polygon.
    """
    poly_np = np.array(poly).reshape(-1, 2)
    x_min, y_min = np.min(poly_np, axis=0)
    x_max, y_max = np.max(poly_np, axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


if __name__ == "__main__":
    root_dir = "data/train"
    all_folders = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
    ])

    # Split into train/val
    train_folders, val_folders = train_test_split(all_folders, test_size=0.1, random_state=42)

    print(f"Train: {len(train_folders)} folders, Val: {len(val_folders)} folders")

    # Generate train.json and val.json
    last_img_id, last_ann_id = generate_coco_json(
        train_folders, root_dir, "train.json"
    )
    generate_coco_json(
        val_folders, root_dir, "val.json",
        start_image_id=last_img_id,
        start_annotation_id=last_ann_id
    )

    # ✅ Generate full.json (all folders)
    print(f"Generating full.json from {len(all_folders)} folders...")
    generate_coco_json(all_folders, root_dir, "full.json")
