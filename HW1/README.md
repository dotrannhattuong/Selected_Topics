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
