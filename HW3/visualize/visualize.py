import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load COCO annotation file
coco_path = "/mnt/HDD1/tuong/selected/data/all.json"
with open(coco_path, "r") as f:
    coco = json.load(f)

# Extract bounding box dimensions
widths, heights, areas = [], [], []
for ann in coco["annotations"]:
    x, y, w, h = ann["bbox"]
    widths.append(w)
    heights.append(h)
    areas.append(w * h)

# === Plot histograms ===
plt.figure(figsize=(24, 6))  # Bigger figure

# Width distribution
plt.subplot(1, 3, 1)
sns.histplot(widths, bins=50, kde=True)
plt.title("Width Distribution", fontsize=30)
plt.xlabel("Width (pixels)", fontsize=28)
plt.ylabel("Count", fontsize=28)
plt.tick_params(axis='both', labelsize=26)

# Height distribution
plt.subplot(1, 3, 2)
sns.histplot(heights, bins=50, kde=True)
plt.title("Height Distribution", fontsize=30)
plt.xlabel("Height (pixels)", fontsize=28)
plt.ylabel("Count", fontsize=28)
plt.tick_params(axis='both', labelsize=26)

# Area distribution (log scale)
plt.subplot(1, 3, 3)
sns.histplot(areas, bins=50, kde=True)
plt.title("Area Distribution (log scale)", fontsize=30)
plt.xlabel("Area (w × h)", fontsize=28)
plt.ylabel("Count", fontsize=28)
plt.yscale("log")
plt.tick_params(axis='both', labelsize=26)

# Save figure
plt.tight_layout()
plt.savefig("object_size_distribution.pdf", format="pdf")
plt.close()

# === Print statistics ===
median_width = np.median(widths)
median_height = np.median(heights)
median_area = np.median(areas)
small_objects = [a for a in areas if a <= 32 * 32]
small_obj_percent = len(small_objects) / len(areas) * 100

print(f"Median Width: {median_width:.1f}")
print(f"Median Height: {median_height:.1f}")
print(f"Median Area: {median_area:.1f}")
print(f"Small Objects (area ≤ 1024 px²): {len(small_objects)} / {len(areas)} ({small_obj_percent:.2f}%)")
