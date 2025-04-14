#!/usr/bin/env python3
"""
Plot bounding box statistics from a COCO-format JSON file and save each plot as a PDF file.

This script loads a JSON file (e.g., train.json) with COCO dataset annotations,
extracts the bounding box widths, heights, and areas, and then uses matplotlib to plot
separate histograms for each statistic. The resulting plots are saved as separate PDF files:
    - width_distribution.pdf
    - height_distribution.pdf
    - area_distribution.pdf

In each plot the tick labels on the x- and y-axes are scaled by 1e3 so that
the values are shown in thousands (×10³). The font size for the scale ("1e3" or similar)
is increased.
"""

import os
import json
import matplotlib.pyplot as plt

# Set the global font to Times New Roman (if available)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"]

# Define constants for the JSON file path and output directory.
JSON_PATH = "/mnt/HDD1/tuong/selected/Selected_Topics/HW2/data/train.json"
OUTPUT_DIR = "visualize/plots"

# Create the output directory if it doesn't exist.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the COCO-format JSON data.
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract annotations (each annotation has a bbox in the form [x, y, width, height]).
annotations = data.get("annotations", [])

# Create lists to store widths, heights, and areas.
widths = []
heights = []
areas = []

# Loop through each annotation and compute statistics.
for annot in annotations:
    bbox = annot.get("bbox", [])
    if len(bbox) == 4:
        width = bbox[2]
        height = bbox[3]
        widths.append(width)
        heights.append(height)
        areas.append(width * height)

# Define a helper function to plot and save a histogram.
def plot_and_save(data_list, bins, title, xlabel, ylabel, filename, color='blue'):
    """
    Plot a histogram for the given data and save it as a PDF.

    Args:
        data_list (list): Data values to plot.
        bins (int): Number of bins in the histogram.
        title (str): Title of the plot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        filename (str): Filename (with path) to save the figure.
        color (str): Histogram color.
    """
    plt.figure(figsize=(8, 8))
    plt.hist(data_list, bins=bins, color=color, alpha=0.7)
    plt.title(title, fontsize=22)
    plt.xlabel(xlabel + " (×10³)", fontsize=20)
    plt.ylabel(ylabel + " (×10³)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    ax = plt.gca()
    # Force scientific notation with scale factor around 1e3.
    ax.ticklabel_format(axis='both', style='sci', scilimits=(3, 3))
    # Increase the font size of the offset text (the part that shows "e3" or similar)
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.yaxis.get_offset_text().set_fontsize(20)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    plt.close()

# Plot and save the width distribution.
width_pdf = os.path.join(OUTPUT_DIR, "width_distribution.pdf")
plot_and_save(
    widths,
    bins=50,
    title="Bounding Box Width Distribution",
    xlabel="Width (pixels)",
    ylabel="Frequency",
    filename=width_pdf,
    color='blue'
)

# Plot and save the height distribution.
height_pdf = os.path.join(OUTPUT_DIR, "height_distribution.pdf")
plot_and_save(
    heights,
    bins=50,
    title="Bounding Box Height Distribution",
    xlabel="Height (pixels)",
    ylabel="Frequency",
    filename=height_pdf,
    color='green'
)

# Plot and save the area distribution.
area_pdf = os.path.join(OUTPUT_DIR, "area_distribution.pdf")
plot_and_save(
    areas,
    bins=50,
    title="Bounding Box Area Distribution",
    xlabel="Area (pixels\u00b2)",
    ylabel="Frequency",
    filename=area_pdf,
    color='red'
)

print("PDF files have been saved:")
print(f" - {width_pdf}")
print(f" - {height_pdf}")
print(f" - {area_pdf}")
