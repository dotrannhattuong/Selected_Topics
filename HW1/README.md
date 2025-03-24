# NYCU Computer Vision 2025 Spring HW1

**StudentID:** 312540013  
**Name:** Do Tran Nhat Tuong

---

## Introduction
```
Goal: Classify 100 categories from a 21,024-image dataset, with some classes highly similar (e.g., “white flower”).

Constraints: No external data, model under 100M parameters, only ResNet-based backbones allowed.

Approach: Use ResNet variants, address class imbalance via probability sampling, and apply weak/strong data augmentations.

Enhancements: Integrate attention and Squeeze-and-Excitation (SE) modules to improve feature extraction.

Experiments: Evaluate different batch sizes, learning rate schedulers, and optimizers, reporting comprehensive performance results.
```

---

## How to install
```
git clone https://github.com/dotrannhattuong/Selected_Topics.git
cd Selected_Topics/HW1
conda env create -f environment.yml
```

## Folder Format:
```
HW1
-augmentation
-data
    |-train
    |-val
    |-test
-log_utils
-utils
-visualize
```

## Training
```
cd Selected_Topics/HW1
# Best (SE): python train.py
# Attention: python train_att.py
# Weighted imbalance: python train_imbl.py
# Training with the augmentation data: python train_aug.py
```

## Appendix
### Offline augmentation
```
cd Selected_Topics/HW1
# Weak: python augmentation/weak.py
# Strong: python augmentation/strong.py
```

### Visualization
```
cd Selected_Topics
# Training Curve: python -m HW1.visualize.training_curve
# Confusion Matrix: python -m custom.visualize.corre
# Count number of classes: python -m HW1.visualize.num_class
```
