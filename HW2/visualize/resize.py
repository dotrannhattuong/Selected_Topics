#!/usr/bin/env python3
"""
Randomly select 5 images from each of three folders (data/train, data/valid, data/test) 
and resize the selected images to a square of a specified size.
The resized images are saved in an output folder ("resized_samples").

Requirements:
  - OpenCV (cv2)
  - Python 3.x

Usage:
  $ python resize_images_to_square.py
"""

import os
import glob
import random
import cv2

# Folders to sample images from.
FOLDERS = ["data/train", "data/valid", "data/test"]

# Output folder to save resized images.
OUTPUT_DIR = "visualize/resized_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set the desired square target size (width and height in pixels).
TARGET_SIZE = 256

# File extensions to consider (case-insensitive).
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Loop over each folder.
for folder in FOLDERS:
    # Get the list of image file paths for valid extensions.
    image_paths = glob.glob(os.path.join(folder, "*"))
    image_paths = [p for p in image_paths if p.lower().endswith(IMG_EXTENSIONS)]
    
    # If there are fewer than 5 images, print a warning and skip this folder.
    if len(image_paths) < 5:
        print(f"Warning: Not enough images in {folder} to sample 5 (found {len(image_paths)}). Skipping.")
        continue
    
    # Randomly select 5 images.
    sample_paths = random.sample(image_paths, 5)
    
    for img_path in sample_paths:
        # Read the image.
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}.")
            continue
        
        # Resize image to a square of dimensions TARGET_SIZE x TARGET_SIZE.
        # Note: this forces the image to be square by ignoring the original aspect ratio.
        resized_img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
        
        # Create an output filename; prefix with the folder name.
        base_name = os.path.basename(img_path)
        folder_name = os.path.basename(folder)
        out_name = f"{folder_name}_{base_name}"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # Save the resized image.
        cv2.imwrite(out_path, resized_img)
        print(f"Saved resized image to {out_path}")

print("Resizing complete.")
