import os
import json
import cv2
import argparse
import zipfile
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from mpvit import add_mpvit_config

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_folder", type=str, default="../../data/test_release")
    parser.add_argument("--output_dir", type=str, default="results/mask_rcnn_mpvit_base_ms_3x")
    parser.add_argument("--trained_model", type=str, default="output/mask_rcnn_mpvit_base_ms_3x")
    parser.add_argument("--mapping_json", type=str, default="../../data/test_image_name_to_ids.json")
    parser.add_argument("--device", type=str, default="cuda")
    return parser

def setup_model(args):
    cfg = get_cfg()
    add_mpvit_config(cfg)
    cfg.merge_from_file(os.path.join(args.trained_model, "config.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(args.trained_model, "model_best.pth")
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000    
    return DefaultPredictor(cfg)

def load_image_id_mapping(mapping_json_path):
    with open(mapping_json_path, "r") as f:
        mapping_list = json.load(f)
    name_to_id = {item["file_name"]: item["id"] for item in mapping_list}
    size_dict = {item["file_name"]: (item["height"], item["width"]) for item in mapping_list}
    return name_to_id, size_dict

def main():
    args = get_parser().parse_args()
    predictor = setup_model(args)

    os.makedirs(args.output_dir, exist_ok=True)
    visualize_dir = os.path.join(args.output_dir, "visualize")
    os.makedirs(visualize_dir, exist_ok=True)

    json_output_path = os.path.join(args.output_dir, "test-results.json")
    name_out_dir = os.path.basename(args.output_dir)
    zip_output_path = os.path.join(args.output_dir, f"{name_out_dir}.zip")

    name_to_id, size_dict = load_image_id_mapping(args.mapping_json)
    test_files = sorted(f for f in os.listdir(args.test_folder) if f.lower().endswith((".png", ".tif")))

    results = []

    for file_name in tqdm(test_files, desc="Instance prediction"):
        if file_name not in name_to_id:
            print(f"Skipping: {file_name}")
            continue

        image_id = name_to_id[file_name]
        H, W = size_dict[file_name]
        image_path = os.path.join(args.test_folder, file_name)

        image = cv2.imread(str(image_path))
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")

        # Save visualize
        visualizer = Visualizer(image[:, :, ::-1], scale=0.5)
        vis_output = visualizer.draw_instance_predictions(instances)
        out_path = os.path.join(visualize_dir, file_name.replace(".tif", ".png"))
        cv2.imwrite(out_path, vis_output.get_image()[:, :, ::-1])

        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        for box, mask, score, cls in zip(boxes, masks, scores, classes):
            bbox = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")

            results.append({
                'image_id': image_id,
                'bbox': bbox,
                'score': float(score),
                'category_id': int(cls) + 1,
                'segmentation': {
                    "size": [H, W],
                    "counts": rle["counts"]
                }
            })

    # Sort results by image_id
    results = sorted(results, key=lambda x: x["image_id"])

    # Save JSON
    with open(json_output_path, "w") as f:
        json.dump(results, f)
    print(f"\n✅ JSON saved to: {json_output_path}")

    # Save ZIP
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_output_path, arcname=os.path.basename(json_output_path))
    print(f"✅ ZIP archive saved to: {zip_output_path}")

if __name__ == "__main__":
    main()
