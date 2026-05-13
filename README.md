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

## 📊 Results

### Validation set (MNIST, 10% holdout)

| Model | Accuracy | F1 Score (macro) |
|---|---|---|
| KNN |98% |0.98 |
| SVM |99% |0.99 |
| Decision tree |87% |0.87 | 
| Logistic regression |92% |0.91 |
**SVM ON TEST SET** Accuraacy: 98%, F1-score(macro): 0.98 

### Hand-Drawn digits (custom PNG)

| Model | Accuracy | F1 Score (macro) |
|---|---|---|
| KNN |79% |0.76 |
| SVM |85% |0.84 |
| Decision tree |55% |0.50 |
| Logistic regression |81% |0.79 |

> **Why the drop on hand-drawn digits is expected - and actually valuable.**
>
> All models are trained on MNIST, where digits are centered, normalized, and written in a fairly uniform style. My hand-drawn inputs introduce real-world variability: personal handwriting style, stroke thickness, angle, and spacing all differ from the training distribution. This gap is a well-known challenge in applied ML - a model that scores 97% on a clean benchmark can struggle on data it has never seen before.
>
> Testing on custom drawings makes this project more honest than a standard MNIST benchmark. It shows where the model generalizes and where it doesn't, which is exactly the kind of analysis that matters outside of Kaggle.

---

## Pipeline

```
MNIST PNG dataset (train / test)
        ↓
  Image loading & flattening
  (PIL, NumPy - 28×28 → 784-dim vector)
        ↓
  Train / Validation split
  (90% train, 10% val, stratified)
        ↓
  Model training
  (KNN, SVM, Decision tree, Logistic regression)
        ↓
  Validation comparison
  (accuracy + classification report)
        ↓
  MNIST Test set evaluation (10k images)
        ↓
  Hand-Drawn digit preprocessing
  (RGBA → grayscale → invert → center → resize)
        ↓
  Final evaluation on custom drawings
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

## How to run

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

## 📁 Project structure

```
├── Mnist_final.ipynb      # Full pipeline: training, comparison, custom digit testing
├── drawings/              # My hand-drawn digit images
└── README.md
```
