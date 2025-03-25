# NYCU Computer Vision - Spring 2025: Homework 1

**Student ID:** 312540013  
**Name:** Do Tran Nhat Tuong

---

## 🔍 Overview

**Objective:**  
Classify images into 100 fine-grained categories from a dataset containing 21,024 samples. Some classes exhibit high visual similarity (e.g., different kinds of white flowers), posing additional challenges.

**Constraints:**  
- No use of external datasets  
- Model size < 100M parameters  
- Backbone limited to ResNet variants

**Approach:**  
- Experiment with different ResNet-based architectures  
- Handle class imbalance using probability sampling  
- Apply both weak and strong data augmentations  
- Integrate advanced modules like Attention and Squeeze-and-Excitation (SE) to enhance feature representation  
- Test various training setups (batch sizes, learning rate schedulers, optimizers) and report results

---

## ⚙️ Setup

Clone the repository and create the environment:

```bash
git clone https://github.com/dotrannhattuong/Selected_Topics.git
cd Selected_Topics/HW1
conda env create -f environment.yml
```

---

## 📁 Project Structure

```
HW1/
│
├── augmentation/        # Offline augmentation scripts (weak & strong)
├── data/                # Dataset: train/val/test folders
├── log_utils/           # Logging and monitoring utilities
├── utils/               # Helper functions
├── visualize/           # Visualization scripts
```

---

## 🚀 Training

Run one of the following training scripts:

```bash
cd Selected_Topics/HW1

# Baseline with SE module
python train.py

# Model with attention mechanism
python train_att.py

# Handle class imbalance with weighted sampling
python train_imbl.py

# Train with augmented data
python train_aug.py
```

---

## 🧩 Appendix

### 🔄 Offline Augmentation

Apply data augmentation before training:

```bash
# Weak augmentation
python augmentation/weak.py

# Strong augmentation
python augmentation/strong.py
```

---

### 📊 Visualization

Generate training metrics and analysis:

```bash
cd Selected_Topics

# Plot training/validation loss & accuracy curves
python -m HW1.visualize.training_curve

# Draw confusion matrix
python -m custom.visualize.corre

# Count number of images per class
python -m HW1.visualize.num_class
```

---

### Performance
| Num | Backbone | Parameters (M) | Testing Image Size | Validation Accuracy (%) | Public Testing Accuracy (%) |
|-----|----------|----------------|---------------------|--------------------------|------------------------------|
| 1 | [ResNet50x4](https://huggingface.co/timm/resnet50x4_clip.openai) | 85.75 | 288 | 91.00 | 93.00 |
| 2 | [ResNest101e](https://huggingface.co/timm/resnest101e.in1k) | 46.43 | 256 | 91.67 | 95.00 |
| 3 | [ResNest200e](https://huggingface.co/timm/resnest200e.in1k) | 68.36 | 320 | 92.33 | 95.00 |
| 4 | [ResNet101 CLIP](https://huggingface.co/timm/resnet101_clip.openai) | 55.42 | 224 | 88.67 | 92.00 |
| 5 | [ResNet152d](https://huggingface.co/timm/resnet152d.ra2_in1k) | 58.37 | 256 | 89.00 | 94.00 |
| 6 | [ResNetrs200](https://huggingface.co/timm/resnetrs200.tf_in1k) | 93.37 | 256 | 93.67 | 96.00 |
| 7 | [ResNext101](https://huggingface.co/timm/resnext101_32x8d.fb_swsl_ig1b_ft_in1k) | 86.95 | 224 | 92.67 | 96.00 |
| 8 | [SEResNextaa101d (288)](https://huggingface.co/timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288) | 91.74 | 288 | 94.00 | 96.00 |
| 9 | [SEResNextaa101d (320)](https://huggingface.co/timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288) | 91.74 | 320 | 95.33 | 97.00 |
| 10 | [SEResNextaa101d (320, lr:1e-5)](https://huggingface.co/timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288) | 91.74 | 320 | 96.00 | 98.00 |

---
> 🚀 *Best performance*: `SEResNextaa101d @ 320px-lr:1e-5` achieved the highest accuracy on both validation (96.00%) and public testing (98.00%) datasets.
