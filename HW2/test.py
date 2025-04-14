#!/usr/bin/env python3
"""
Convert test image predictions to CSV and save visualization images.

This script uses a Detectron2 model to run inference on all PNG images in a 
specified test folder. For each image, it converts the predicted bounding boxes
(from [x1, y1, x2, y2] to [x, y, width, height]) and stores predictions in a list
in COCO format. These predictions are saved as "pred.json" in the output directory.
The script then groups predictions by image_id and creates a label by concatenating
the raw category IDs (as strings), sorted by the x-coordinate of the bounding boxes.
Since the JSON saves (predicted class - 1), the CSV adjusts them by adding 1.
If an image has no predictions, its label is set to "-1". Visualizations of the 
predictions are saved in a "results" folder. Finally, the pred.json and pred.csv files
are both zipped into one archive.
"""

import os
import json
import csv
import cv2
import argparse
import zipfile
from tqdm import tqdm

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def get_parser():
    """
    Create an argument parser for command-line parameters.
    
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run inference on test images and produce prediction JSON, CSV, and visualizations."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/models/R50-C4_1x"
    )
    parser.add_argument(
        "--test-folder",
        type=str,
        default="data/test",
        help="Folder containing test images (default: data/test)"
    )
    parser.add_argument(
        "--trained_model",
        type=str,
        default="checkpoints/models/R50-C4_1x"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:5",
        help="Device to use for inference (default: cuda:0)"
    )
    return parser


def setup_model(args):
    """
    Set up and return a Detectron2 DefaultPredictor using parameters provided in args.
    
    Args:
        args (Namespace): Parsed command-line arguments.
    
    Returns:
        DefaultPredictor: The configured predictor.
    """
    cfg = get_cfg()
    cfg_path = os.path.join(args.trained_model, "config.yaml")
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = os.path.join(args.trained_model, "model_best.pth")
    dataset_name = "my_dataset_test"
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # e.g. digit detection: 10 classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()
    return DefaultPredictor(cfg)


def group_predictions(predictions):
    """
    Group predictions by image ID.
    
    Args:
        predictions (list): List of prediction dictionaries.
    
    Returns:
        dict: Dictionary mapping image_id to list of (x-coordinate, category_id as string) tuples.
    """
    pred_dict = {}
    for pred in predictions:
        image_id = pred["image_id"]
        x_left = pred["bbox"][0]  # x-coordinate of the bounding box
        # The JSON was saved with "predicted class - 1"
        category_str = str(pred["category_id"])
        pred_dict.setdefault(image_id, []).append((x_left, category_str))
    return pred_dict


def create_pred_label(pred_list):
    """
    Sort the prediction list by x-coordinate and concatenate adjusted category IDs.
    
    Since the JSON stores (predicted class - 1), we add 1 to each value to recover
    the original digit string.
    
    Args:
        pred_list (list): A list of tuples (x-coordinate, category_id as string).
    
    Returns:
        str: The concatenated digit string.
    """
    sorted_preds = sorted(pred_list, key=lambda tup: tup[0])
    return "".join(str(int(category) - 1) for _, category in sorted_preds)


def main():
    """Main function: runs inference on test images, saves predictions, CSV, and ZIP archives."""
    parser = get_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    test_folder = args.test_folder
    results_folder = os.path.join(output_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    predictor = setup_model(args)
    test_metadata = MetadataCatalog.get("my_dataset_test")

    # List PNG files in the test folder and sort them numerically by filename.
    test_files = sorted(
        [f for f in os.listdir(test_folder) if f.lower().endswith(".png")],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    all_image_ids = set()
    coco_predictions = []

    for filename in tqdm(test_files, desc="Processing images"):
        try:
            image_id = int(os.path.splitext(filename)[0])
        except ValueError:
            print(f"Skipping file with non-integer name: {filename}")
            continue
        all_image_ids.add(image_id)
        image_path = os.path.join(test_folder, filename)
        im = cv2.imread(image_path)
        if im is None:
            print(f"Warning: could not read image {image_path}")
            continue

        # Run inference on the image.
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")

        # Visualize predictions on the image.
        v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.8)
        out = v.draw_instance_predictions(instances)
        output_image_path = os.path.join(results_folder, filename)
        cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])

        # If there are no predictions, skip appending any detection.
        if len(instances) == 0:
            continue

        # Get predicted boxes, scores, and classes.
        boxes = instances.pred_boxes.tensor.tolist()  # boxes in the format [x1, y1, x2, y2]
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        for bbox, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            # Save the prediction in COCO format. Note that we subtract 1 from class IDs.
            coco_predictions.append({
                "image_id": image_id,
                "category_id": int(cls)+1,
                "score": score,
                "bbox": [x1, y1, width, height]
            })

    # Save the generated predictions to "pred.json".
    output_json_path = os.path.join(output_dir, "pred.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_predictions, f, indent=4)
    print(f"Saved predictions to {output_json_path}.")

    # Group predictions by image_id.
    pred_dict = group_predictions(coco_predictions)

    # Create prediction labels for the CSV output.
    results = []
    for image_id in sorted(all_image_ids):
        if image_id not in pred_dict or not pred_dict[image_id]:
            pred_label = "-1"
        else:
            pred_label = create_pred_label(pred_dict[image_id])
        results.append((image_id, pred_label))

    # Write the results to "pred.csv".
    output_csv_path = os.path.join(output_dir, "pred.csv")
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "pred_label"])
        for image_id, label in results:
            writer.writerow([image_id, label])
    print(f"CSV file created at {output_csv_path} with {len(results)} entries.")

    # Zip the pred.json and pred.csv files into one archive.
    zip_task = os.path.join(output_dir, f"{os.path.basename(output_dir)}.zip")
    with zipfile.ZipFile(zip_task, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_json_path, arcname="pred.json")
        zipf.write(output_csv_path, arcname="pred.csv")
    print(f"Zipped pred.json and pred.csv to {zip_task}.")
    print(f"Visualized images have been saved in the folder '{results_folder}'.")


if __name__ == "__main__":
    main()
