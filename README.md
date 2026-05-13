#  MNIST Digit classifier - ML model comparison

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-yellowgreen.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

Training and comparing four ML models on the MNIST dataset, then testing them on **self-drawn handwritten digits**.

---

## 📌 Project Overview

MNIST is a classic benchmark of 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels. This project goes beyond the standard benchmark - after training, all models are evaluated on digits drawn by hand and preprocessed to match the MNIST format.

---

## 🤖 Models Compared

| Model | Notes |
|---|---|
| KNN | k=3, parallel jobs |
| SVM | RBF kernel, C=5, gamma=scale |
| Decision Tree | Default depth, random_state=42 |
| Logistic Regression | max_iter=1000 |

> **Best performer: SVM** - highest accuracy on both the MNIST test set and hand-drawn digits.

---

## Pipeline

```
MNIST PNG Dataset (train / test)
        ↓
  Image Loading & Flattening
  (PIL, NumPy — 28×28 → 784-dim vector)
        ↓
  Train / Validation Split
  (90% train, 10% val, stratified)
        ↓
  Model Training
  (KNN, SVM, Decision Tree, Logistic Regression)
        ↓
  Validation Comparison
  (accuracy + classification report)
        ↓
  MNIST Test Set Evaluation (10k images)
        ↓
  Hand-Drawn Digit Preprocessing
  (RGBA → grayscale → invert → center → resize)
        ↓
  Final Evaluation on Custom Drawings
```

---

## Hand-Drawn digit testing

One of the highlights of this project. Digits drawn on a white background are preprocessed to match MNIST format (white digit on black background), centered, and resized to 28×28. All four models are then evaluated on these custom inputs.

Preprocessing steps:
- Convert RGBA → alpha channel extraction
- Threshold noise (alpha < 30 → 0)
- Crop to bounding box
- Center of mass alignment via `scipy.ndimage`
- Resize to 28×28

---

## How to Run

### 1. Download the dataset

👉 [MNIST PNG format on Kaggle](https://www.kaggle.com/datasets/jidhumohan/mnist-png)

Extract it so the folder structure looks like:

```
mnist_png/
├── train/
│   ├── 0/
│   ├── 1/
│   └── ... (up to 9)
└── test/
    ├── 0/
    └── ...
```

### 2. Update the path in the notebook

In Cell 0, change `train_path` to your local folder:

```python
train_path = "/your/path/to/mnist_png/train/"
```

### 3. Install dependencies

```bash
pip install numpy pillow scikit-learn matplotlib scipy
```

### 4. Run the notebook

```bash
jupyter notebook Mnist_final.ipynb
```

---

## 📁 Project Structure

```
├── Mnist_final.ipynb      # Full pipeline: training, comparison, custom digit testing
├── drawings/              # My hand-drawn digit images
└── README.md
```
