#!/usr/bin/env python3
"""
Plot bounding box statistics from a COCO-format JSON file and save each plot as a 
PDF file.

This script loads a JSON file (e.g., train.json) containing COCO dataset annotations,
extracts the bounding box widths, heights, and areas, and then uses matplotlib to plot 
separate histograms for these statistics. The resulting plots are saved as three 
different PDF files: width_distribution.pdf, height_distribution.pdf, and 
area_distribution.pdf.

The tick labels on the y-axis are scaled by 1e3 (i.e., ×10³).
"""

import os
import json
import matplotlib.pyplot as plt

# Set the global font to Times New Roman (if available)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"
]

# Define constants for the JSON file path and output directory.
COCO_JSON_PATH = (
    "/mnt/HDD1/tuong/selected/Selected_Topics/HW2/data/train.json"
)
OUTPUT_DIR = "visualize/plots"

# Create the output directory if it does not exist.
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_aspect_ratio_histogram(json_file_path):
    """
    Read a COCO-format JSON file and plot a histogram of bounding box aspect ratios.

    Args:
        json_file_path (str): Path to the COCO JSON file (e.g., train.json).
    """
    # Open the JSON file and load its data.
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract bounding box annotations from the key "annotations".
    # Each bounding box is assumed to be in the format [x, y, width, height].
    aspect_ratios = []
    for annotation in data.get("annotations", []):
        bbox = annotation.get("bbox", [])
        if len(bbox) == 4 and bbox[3] > 0:
            width = bbox[2]
            height = bbox[3]
            aspect_ratio = width / height
            aspect_ratios.append(aspect_ratio)

    # Print the total number of bounding boxes processed.
    print(f"Total number of bounding boxes: {len(aspect_ratios)}")

    # Plot the histogram of aspect ratios.
    plt.figure(figsize=(10, 6))
    plt.hist(aspect_ratios, bins=30, edgecolor="black", color="skyblue")
    plt.xlabel("Aspect Ratio (width / height)", fontsize=20)
    plt.ylabel("Count (×10³)", fontsize=20)
    plt.title("Histogram of Bounding Box Aspect Ratios", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    
    # Scale tick labels by 1e3 for the y-axis only.
    ax = plt.gca()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    
    plt.tight_layout()
    
    # Save the plot as a PDF.
    out_pdf = os.path.join(OUTPUT_DIR, "aspect_ratio_distribution.pdf")
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    
    print(f"Aspect ratio histogram saved to {out_pdf}")


if __name__ == "__main__":
    plot_aspect_ratio_histogram(COCO_JSON_PATH)
