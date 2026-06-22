# E-Commerce Purchase Intention Prediction

> **A Comparative Analysis of Linear and Non-Linear Models for E-Commerce Purchase Prediction**

Comparative study of **Logistic Regression**, **Artificial Neural Network (ANN)**, and **Random Forest** for predicting whether an online shopping session will result in a purchase (binary classification: `Revenue = True/False`).

**Author:** Davide Meta \
**Course:** Intelligenza Artificiale e Apprendimento Automatico — Prof. Federico Pernici \
**University:** Università degli Studi di Firenze — Dipartimento di Ingegneria dell'Informazione

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Methodology](#methodology)
- [Key Design Decisions](#key-design-decisions)
- [Results](#results)
- [References](#references)
- [Reproducibility](#reproducibility)

---

## Project Overview

This project tackles a **binary classification** problem: predicting whether a web session on an e-commerce platform will result in a purchase (`Revenue = True`) or not (`Revenue = False`). The dataset is highly **imbalanced** (~84.5% negative, ~15.5% positive), which drives many of the design choices documented below.

Three models are trained and compared:

| Model | Type | Role |
|-------|------|------|
| Logistic Regression | Linear | Baseline / interpretable benchmark |
| ANN (2 hidden layers) | Deep Learning | Non-linear feature interactions |
| Random Forest (500 trees) | Ensemble | Robust non-linear + SHAP interpretability |

---

## Dataset Description

**Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

- **Sessions**: 12,330
- **Numerical features (10)**: Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay
- **Categorical features (8)**: Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend, Revenue (target)
- **Target**: `Revenue` — `True` (~15.5%) / `False` (~84.5%)

---

## Project Structure

```
ecommerce-intention/
├── data/
│   ├── online_shoppers_intention.csv       # Original dataset
│   └── preprocessed_data.npz               # Preprocessed arrays
├── notebooks/
│   ├── 01_eda.ipynb                        # Exploratory Data 
│   ├── 02_preprocessing.ipynb              # Preprocessing 
│   ├── 03_logistic_regression.ipynb        # Logistic Regression 
│   ├── 04_ann.ipynb                        # ANN training & 
│   ├── 05_random_forest.ipynb              # Random Forest + 
│   └── 06_evaluation_comparison.ipynb      # Cross-model 
├── report/
│   ├── Part/                               # LaTeX source files
│   └── Resources/                          # Figures and images
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                    # Data loading, encoding, scaling, SMOTE
│   ├── models.py                           # Model definitions and training
│   └── evaluation.py                       # Metrics, plots, SHAP analysis
├── .gitignore
├── README.md
├── report.pdf                              # Compiled project report
└── requirements.txt
```

---

## Setup & Installation

```bash
# 1. Clone or download the repository
cd ecommerce-intention

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and run notebooks sequentially
jupyter notebook notebooks/
```

> **Important**: Run notebooks in order (01 → 06). Notebook 02 generates the `preprocessed_data.npz` file that notebooks 03–06 depend on.

---

## Methodology

### Preprocessing Pipeline

1. **Feature Encoding**:
   - Ordinal encoding for `Month` (Jan=1 … Dec=12)
   - Label encoding for `VisitorType` (Returning=0, New=1, Other=2)
   - Boolean → integer for `Weekend` and `Revenue`
2. **Train/Test Split**: 80/20, stratified on target, `random_state=42`
3. **Standard Scaling**: `StandardScaler` fit on training set only, applied to both sets
4. **SMOTE Oversampling**: Applied exclusively to the training set (never on test data)

### Models

| Model | Architecture | Class Imbalance Strategy |
|-------|-------------|--------------------------|
| **Logistic Regression** | `solver='lbfgs'`, `max_iter=1000` | `class_weight='balanced'` |
| **ANN** | 128 → BN → Dropout(0.3) → 64 → BN → Dropout(0.3) → 1, ReLU + Sigmoid output, Adam, EarlyStopping | `class_weight={0: 1.0, 1: 5.0}` |
| **Random Forest** | 500 estimators, `n_jobs=-1` | `class_weight='balanced'` |

### Evaluation Suite

Each model is evaluated with the full suite:

- **Confusion Matrix** (heatmap)
- **Classification Report** (Precision, Recall, F1-Score per class)
- **ROC Curve + AUC Score** (individual per model)
- **Comparative ROC** (all three models overlaid in notebook 06)
- **SHAP Analysis** (beeswarm + bar plots for Random Forest)

---

## Key Design Decisions

### Why SMOTE Is Applied After the Train/Test Split

SMOTE (Synthetic Minority Oversampling Technique) generates synthetic samples by interpolating between existing minority-class instances. If applied **before** the split:

- Synthetic samples in the **test set** may be derived from (or identical to) training samples
- This constitutes **data leakage**: the model's evaluation becomes overly optimistic because the test set no longer represents unseen real-world data
- The test set's class distribution would be artificially balanced, misrepresenting the production environment

**Correct pipeline**: Split → SMOTE on train only → Train model → Evaluate on original (imbalanced) test set.

### Why ReLU Is Preferred Over Sigmoid in Hidden Layers

The **Sigmoid** activation function squashes all outputs into the range [0, 1]. In deep networks, this causes the **vanishing gradient problem**:

- During backpropagation, gradients are multiplied across layers
- Sigmoid's maximum gradient is 0.25 (at x = 0), so gradients shrink exponentially through each layer
- In networks with 2+ hidden layers, early layers receive near-zero gradients and stop learning

**ReLU** (`max(0, x)`) solves this:

- Gradient is **1** for all positive inputs — no attenuation
- Computationally cheaper (no exponentials)
- Sigmoid remains appropriate for the **output layer** in binary classification, where a [0, 1] probability is needed

### Why ROC-AUC Complements F1-Score for Imbalanced Datasets

| Metric | What It Measures | Threshold-Dependent? | Class-Distribution Sensitive? |
|--------|-----------------|---------------------|-------------------------------|
| F1-Score | Harmonic mean of Precision & Recall | Yes (default = 0.5) | Yes |
| ROC-AUC | Area under the TPR vs. FPR curve | No (evaluates all thresholds) | No |

- **F1-Score** is useful for evaluating the model at a specific operating point (threshold), especially when the costs of false positives and false negatives differ
- **ROC-AUC** captures the model's overall **discriminative ability** across all possible thresholds, making it ideal for comparing models regardless of class imbalance
- Together, they provide a complete picture: AUC for ranking ability, F1 for real-world deployment performance

### Why SHAP Is Preferred Over Default Feature Importance

Scikit-learn's default `feature_importances_` (Mean Decrease in Impurity / Gini importance) has known limitations:

1. **Biased toward high-cardinality features**: Features with more unique values (e.g., continuous features) receive inflated importance scores
2. **No directionality**: Gini importance shows *how much* a feature matters, but not *how* — does a higher value push the prediction toward purchase or away from it?
3. **Inconsistency**: Can assign non-zero importance to completely random features

**SHAP** (SHapley Additive exPlanations) addresses all three:

- Based on **Shapley values** from cooperative game theory — mathematically guaranteed to be fair and consistent
- Provides both **magnitude and direction** of each feature's contribution per prediction
- The `summary_plot` shows the distribution of feature effects across the entire dataset
- `TreeExplainer` is optimized for tree-based models, making it computationally feasible for Random Forests

---

## Results

All metrics for Precision, Recall, and F1 refer to the **positive class** (Purchase). Models were evaluated on the original imbalanced test set.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8593 | 0.5348 | 0.7042 | 0.6079 | 0.8801 |
| ANN | 0.8528 | 0.5174 | **0.7408** | 0.6093 | 0.8946 |
| Random Forest | **0.8897** | **0.6297** | 0.6990 | **0.6625** | **0.9217** |

### Key Findings

- **Random Forest** achieves the best overall performance across all primary metrics (ROC-AUC = 0.922, F1 = 0.663), confirming that ensemble tree-based methods are highly effective for structured tabular data.
- **ANN** yields the highest **recall** (0.741), capturing the most actual purchasers — useful when the cost of missing a buyer outweighs the cost of a false positive.
- **Logistic Regression** proves remarkably competitive (F1 = 0.608 vs ANN's 0.609), illustrating **Occam's Razor**: much of the predictive signal in this dataset is linearly separable.
- **PageValues** emerged as the overwhelmingly dominant predictor via SHAP analysis, providing a directly actionable business insight.

---

## References

1. **Dataset**: Sakar, C.O., Polat, S.O., Katircioglu, M., & Kastro, Y. (2019). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. [Link](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

2. **SHAP**: Lundberg, S.M., & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS. [Documentation](https://shap.readthedocs.io/)

3. **SMOTE**: Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 16, 321-357.

4. **Random Forest**: Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

5. **Tree-based models on tabular data**: Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS.

---

## Reproducibility

- All random states are fixed to `seed=42`
- File paths use `pathlib.Path` for cross-platform compatibility
- Each notebook is self-contained and runnable top-to-bottom in sequential order (01 → 06)
- Python 3.11+ required; all dependency versions are pinned in `requirements.txt`

---

*Project for Intelligenza Artificiale e Apprendimento Automatico — Università degli Studi di Firenze, A.Y. 2025/2026*
