import os
import cv2
import random
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Input root directory (contains 'lr/' and 'hr/' subfolders)
ROOT_DIR = "videogames_data/train"
LR_DIR = os.path.join(ROOT_DIR, "lr")
HR_DIR = os.path.join(ROOT_DIR, "hr")

# Output base directory for paired patches
OUTPUT_BASE = "datasets"
TRAIN_LR_OUT = os.path.join(OUTPUT_BASE, "train", "LR")
TRAIN_HR_OUT = os.path.join(OUTPUT_BASE, "train", "HR")
VAL_LR_OUT = os.path.join(OUTPUT_BASE, "val", "LR")
VAL_HR_OUT = os.path.join(OUTPUT_BASE, "val", "HR")

# Desired number of patches
NUM_TRAIN = 64935
NUM_VAL = 7220

# Each source image will produce up to 5 random crops
CROPS_PER_IMAGE = 5

# Patch sizes
LR_PATCH_SIZE = 64
HR_PATCH_SIZE = LR_PATCH_SIZE * 4  # 256

# ─── PREPARE OUTPUT DIRECTORIES ───────────────────────────────────────────────

for path in [TRAIN_LR_OUT, TRAIN_HR_OUT, VAL_LR_OUT, VAL_HR_OUT]:
    os.makedirs(path, exist_ok=True)

# ─── GATHER LR FILENAMES AND PRECOMPUTE DIMENSIONS ─────────────────────────────

lr_filenames = [
    f for f in os.listdir(LR_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
assert lr_filenames, f"No LR images found in {LR_DIR}"

# Precompute dimensions for each LR filename: (w_lr, h_lr)
dims = {}
for fname in lr_filenames:
    lr_path = os.path.join(LR_DIR, fname)
    lr_img = cv2.imread(lr_path)
    if lr_img is None:
        continue
    h_lr, w_lr = lr_img.shape[:2]
    dims[fname] = (w_lr, h_lr)

# ─── FUNCTION TO GENERATE MULTIPLE PATCHES FOR ONE IMAGE ──────────────────────

def generate_crops_for_image(lr_fname: str, train_count: int, val_count: int):
    """
    Generate up to CROPS_PER_IMAGE random crops from a single LR/HR pair.
    Save to train or val folders until quotas are met.

    Returns updated (train_count, val_count, saved_this_image).
    """
    w_lr, h_lr = dims[lr_fname]
    base_name = os.path.splitext(lr_fname)[0]  # e.g. "00011"
    
    lr_path = os.path.join(LR_DIR, lr_fname)
    hr_path = os.path.join(HR_DIR, lr_fname)  # assume same filename for HR

    lr_img = cv2.imread(lr_path)
    hr_img = cv2.imread(hr_path)

    saved_this_image = 0

    for k in range(CROPS_PER_IMAGE):
        # If both train and val quotas are filled, stop
        if train_count >= NUM_TRAIN and val_count >= NUM_VAL:
            break

        # Random top-left corner for LR crop
        max_x_lr = w_lr - LR_PATCH_SIZE
        max_y_lr = h_lr - LR_PATCH_SIZE
        x_lr = random.randint(0, max_x_lr)
        y_lr = random.randint(0, max_y_lr)

        # Corresponding HR coordinates (scale factor = 4)
        x_hr = x_lr * 4
        y_hr = y_lr * 4

        # Crop LR and HR
        lr_patch = lr_img[y_lr : y_lr + LR_PATCH_SIZE,
                          x_lr : x_lr + LR_PATCH_SIZE]
        hr_patch = hr_img[y_hr : y_hr + HR_PATCH_SIZE,
                          x_hr : x_hr + HR_PATCH_SIZE]

        # Filenames: "<base_name>_rnd<k>.png"
        out_lr_name = f"{base_name}_rnd{k}.png"
        out_hr_name = f"{base_name}_rnd{k}.png"

        # Save to train or val, depending on quotas
        if train_count < NUM_TRAIN:
            cv2.imwrite(os.path.join(TRAIN_LR_OUT, out_lr_name), lr_patch)
            cv2.imwrite(os.path.join(TRAIN_HR_OUT, out_hr_name), hr_patch)
            train_count += 1
            saved_this_image += 1
        elif val_count < NUM_VAL:
            cv2.imwrite(os.path.join(VAL_LR_OUT, out_lr_name), lr_patch)
            cv2.imwrite(os.path.join(VAL_HR_OUT, out_hr_name), hr_patch)
            val_count += 1
            saved_this_image += 1
        else:
            break

    return train_count, val_count, saved_this_image

# ─── MAIN PATCH GENERATION LOOP ────────────────────────────────────────────────

if __name__ == "__main__":
    train_count = 0
    val_count = 0

    # Shuffle LR filenames so cropping order is random
    random.shuffle(lr_filenames)

    pbar = tqdm(total=NUM_TRAIN + NUM_VAL, desc="Generating Patches")
    
    for lr_fname in lr_filenames:
        if train_count >= NUM_TRAIN and val_count >= NUM_VAL:
            break

        train_count, val_count, saved = generate_crops_for_image(
            lr_fname, train_count, val_count
        )
        pbar.update(saved)

    pbar.close()
    print(f"Done! Generated {train_count} train and {val_count} val paired patches.")
