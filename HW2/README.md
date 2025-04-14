# NYCU Computer Vision - Spring 2025: Homework 2

**Student ID:** 312540013  
**Name:** Do Tran Nhat Tuong

---

## 🔍 Overview

**Objective:**  
Detect house numbers (digits) in images. The goal is to localize and recognize digits from images taken from houses. Because digits often occur in a group (e.g., “123”), post‑processing is applied to concatenate individual digit predictions into a final house number.  

**Constraints:**  
- Use only Faster R-CNN.
- No extended data.
- Model must output both raw bounding box predictions in COCO format and final digit sequences (CSV).

**Approach:**  
- Use a Faster R‑CNN model (e.g., X_101_32x8d_FPN_3x) pre‑trained on COCO, then fine‑tune on the house digits dataset.
- Set the number of classes to 10.
- Apply custom data augmentation (e.g., cropping, resizing) based on analysis of bounding box distributions.
- Tune anchor sizes and aspect ratios using bounding box statistics (e.g., width, height and aspect ratio histograms).
- Post‑process predictions to group adjacent bounding boxes and produce final digit sequence labels.
- Package predictions as two zip files:
  - **Task 1:** Raw COCO predictions (`pred.json`).
  - **Task 2:** Final digit sequence predictions (`pred.csv`).

---

## ⚙️ Setup

Clone the repository and set up your environment as follows:

```bash
conda create -n hw2-env python=3.8
conda activate hw2-env
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python

## 📁 Project Structure
```bash
HW2/
│
├── configs/              # Configuration files for model training.
├── data/                 # COCO-format dataset folders: train, valid, test.
├── logs/                 # Training logs and TensorBoard files.
├── outputs/              # Model checkpoints and prediction outputs.
├── scripts/              # Training, evaluation, and inference scripts.
├── visualize/            # Visualization scripts for plots and analysis.
└── README.md             # This file.
```

## 🚀 Training
```bash
# Default Training
python train_model.py --ims-per-batch 8 --config-file COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml --output-dir checkpoints/models/X101-FPN_3x

# Box Regression Training
python train_bbox.py --ims-per-batch 8 --rpn_box_reg ciou --roi_box_reg ciou --output-dir checkpoints/box_regression/X_ciou_ciou

# Anchor Training 
python train_anchor.py --ims-per-batch 8 --anchor_size 8 16 32 64 128 256 --output-dir checkpoints/anchor/X_8_256
```

## 🧩 Evaluation & Inference
```bash
python test.py --output-dir results/anchor/X_8_256 --trained_model checkpoints/anchor/X_8_256
```

During inference, the following files are produced:

pred.json: Raw COCO-format bounding box predictions.

pred.csv: Post‑processed final digit sequence predictions.

Both files are also packaged into zip files (e.g., *_task1.zip for JSON and *_task2.zip for CSV).

## 📊 Experimental Results


## 📝 Conclusion
In this homework, we fine‑tuned a Faster R‑CNN detector on a custom digit dataset by leveraging targeted data augmentation and loss function adjustments. Our experiments suggest that adjusting anchor sizes and aspect ratios based on the dataset's bounding box statistics—and using Giou loss in the ROI Box Head—can lead to improved localization performance. The results (summarized in the tables above) demonstrate the impact of architectural variations and loss settings on both AP and accuracy.

## 📚 References
- [Detectron2 Documentation](https://detectron2.readthedocs.io/)
- [COCO Dataset Format](https://cocodataset.org/)
- [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)
- [HW2 Presentation Slides](https://docs.google.com/presentation/d/1nrVyofHw3icwmLxEdUTHRZ_uZRAnr02zshddJcvXFfk/edit#slide=id.g33b1fcaa404_0_112)
