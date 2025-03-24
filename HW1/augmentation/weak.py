import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import shutil

# Paths to the original (train) directory and the new augmented directory
TRAIN_DIR = "../data/train"
AUG_DIR = "../data/train_weak"

# Remove the existing augmented directory if it exists, then recreate it
if os.path.exists(AUG_DIR):
    shutil.rmtree(AUG_DIR)
os.makedirs(AUG_DIR)

# Count the number of images in each class folder
class_counts = {}
for class_folder in os.listdir(TRAIN_DIR):
    class_path = os.path.join(TRAIN_DIR, class_folder)
    if os.path.isdir(class_path):
        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        class_counts[class_folder] = len(image_files)

# Find the class with the maximum number of images
max_images = max(class_counts.values())

# Define the augmentation pipeline (sometimes called "strong augmentation")
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.1,
        rotate_limit=30, p=0.7
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.7
    ),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.GridDistortion(p=0.5),
    A.CoarseDropout(
        max_holes=4, max_height=20, max_width=20,
        min_holes=1, min_height=10, min_width=10,
        fill_value=0, p=0.5
    ),
    ToTensorV2()
])

# Create augmented data to balance each class up to 'max_images'
for class_folder in tqdm(class_counts.keys(), desc="Processing Classes"):
    class_path = os.path.join(TRAIN_DIR, class_folder)
    aug_class_path = os.path.join(AUG_DIR, class_folder)
    os.makedirs(aug_class_path, exist_ok=True)

    # List all original images in the class folder
    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]
    num_existing = len(image_files)

    # Copy original images to the new directory
    for img_file in image_files:
        shutil.copy(
            os.path.join(class_path, img_file),
            os.path.join(aug_class_path, img_file)
        )

    # Augment additional images until we reach 'max_images' in this class
    while num_existing < max_images:
        img_name = np.random.choice(image_files)
        img_path = os.path.join(class_path, img_name)

        # Read the image and convert from BGR to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        augmented = augmentation(image=image)['image']

        # Save the new augmented image
        new_img_name = f"aug_{num_existing}_{img_name}"
        cv2.imwrite(
            os.path.join(aug_class_path, new_img_name),
            augmented.permute(1, 2, 0).numpy()
        )

        num_existing += 1

print("Data augmentation completed successfully!")
