# 🔧 Predictive Maintenance Pipeline — AI4I 2020

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A modular, production-ready **Predictive Maintenance (PdM)** machine learning pipeline built with scikit-learn. Uses the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) from Kaggle to predict machine failures before they occur.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Architecture](#-pipeline-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Extending the Pipeline](#-extending-the-pipeline)
- [License](#-license)

---

## 📌 Overview

This project demonstrates a clean, modular ML pipeline design where each step — **cleaning**, **scaling**, and **classification** — is encapsulated in its own Python module and then composed into a single `sklearn.pipeline.Pipeline`. The trained pipeline is serialized to a `.joblib` file for reuse in production inference.

**Key highlights:**
- Fully modular `src/` package: each concern in its own file
- No data leakage: all fitting happens inside the pipeline on training data only
- Handles class imbalance (~97% no-failure) via `class_weight='balanced'`
- Includes cross-validation, evaluation report, and a standalone inference script

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset** — a synthetic dataset reflecting real-world milling machine processes.

| Property | Details |
|---|---|
| Source | [Kaggle — Stephan Matzka](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) |
| Rows | 10,000 |
| Features | 14 |
| Target | `Machine failure` (binary: 0 = Normal, 1 = Failure) |
| Class ratio | ~97% No Failure / ~3% Failure |

**Input features used:**

| Feature | Description |
|---|---|
| `Type` | Machine quality variant: L (Low), M (Medium), H (High) |
| `Air temperature [K]` | Ambient air temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Spindle rotational speed |
| `Torque [Nm]` | Torque applied |
| `Tool wear [min]` | Cumulative tool wear time |

> **Note:** Sub-failure columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded from features — only the binary `Machine failure` label is used as target.

---

## 📁 Project Structure

```
pdm_project/
│
├── train_pipeline.py        ← Entry point: train, evaluate & save pipeline
├── predict.py               ← Inference script for new data samples
├── requirements.txt         ← Python dependencies
│
├── data/
│   └── ai4i2020.csv         ← Place downloaded dataset here
│
├── models/
│   └── pdm_pipeline.joblib  ← Saved pipeline (auto-generated after training)
│
└── src/
    ├── __init__.py          ← Package exports
    ├── cleaner.py           ← DataCleaner transformer
    ├── scaler.py            ← DataScaler transformer
    ├── estimator.py         ← PDMEstimator classifier
    ├── pipeline_builder.py  ← build_pipeline() factory function
    └── evaluate.py          ← evaluate_pipeline() reporting function
```

---

## ⚙️ Pipeline Architecture

The pipeline is composed of three sequential steps:

```
Raw CSV Input
     │
     ▼
┌─────────────┐
│ DataCleaner │  → Drop ID cols, encode 'Type', rename columns, fill NaN
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ DataScaler  │  → StandardScaler on 5 continuous sensor features
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ PDMEstimator │  → XGBoost (scale_pos_weight=33,   # handles imbalance: ~97/3 ratio)
└──────┬───────┘
       │
       ▼
  Prediction (0 = Normal / 1 = Failure)
```

Each step inherits from `sklearn.BaseEstimator` + `TransformerMixin`, making the entire pipeline compatible with `cross_val_score`, `GridSearchCV`, and other sklearn utilities out of the box.

---

## 🚀 Installation

**1. Clone the repository**
```bash
git clone https://github.com/Ditrinov/APML-Streamlit-Project.git
cd pdm-pipeline
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset from Kaggle**

- Go to: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
- Download `ai4i2020.csv`
- Place it in the `data/` folder:
```bash
mv ai4i2020.csv data/
```

---

## 🛠️ Usage

### Training

Run the full training pipeline:

```bash
python train_pipeline.py
```

**What it does:**
1. Loads `data/ai4i2020.csv`
2. Splits into train/test (80/20, stratified)
3. Builds and fits the pipeline (Cleaner → Scaler → Estimator)
4. Runs 5-Fold Cross Validation (ROC-AUC)
5. Evaluates on test set and prints classification report
6. Saves trained pipeline to `models/pdm_pipeline.joblib`

---

### Inference

Predict on new data after training:

```bash
python predict.py
```

Or use in your own script:

```python
import joblib
import pandas as pd

pipeline = joblib.load("models/pdm_pipeline.joblib")

new_sample = {
    "UDI": 9999,
    "Product ID": "M14999",
    "Type": "M",
    "Air temperature [K]": 301.5,
    "Process temperature [K]": 311.0,
    "Rotational speed [rpm]": 1200,
    "Torque [Nm]": 60.0,
    "Tool wear [min]": 220,
}

df = pd.DataFrame([new_sample])
pred  = pipeline.predict(df)[0]
proba = pipeline.predict_proba(df)[0, 1]

print("FAILURE" if pred == 1 else "NORMAL", f"| probability: {proba:.4f}")
```

---

## 📈 Results

Expected performance on the AI4I 2020 dataset:

| Metric | Score |
|---|---|
| ROC-AUC (CV) | ~0.9730 |
| Precision (Failure) | ~0.75 |
| Recall (Failure) | ~0.78 |
| F1-Score (Failure) | ~0.76 |

> Results may vary slightly depending on random seed and scikit-learn version.

---

## 🔄 Extending the Pipeline

### Swap the estimator (e.g., XGBoost)

In `src/estimator.py`, replace `RandomForestClassifier` with:

```python
from xgboost import XGBClassifier

self.model_ = XGBClassifier(
    n_estimators=self.n_estimators,
    max_depth=self.max_depth,
    scale_pos_weight=33,   # handles imbalance: ~97/3 ratio
    random_state=self.random_state,
    eval_metric="logloss",
)
```

### Hyperparameter tuning

```python
from sklearn.model_selection import GridSearchCV
from src.pipeline_builder import build_pipeline

pipeline = build_pipeline()

param_grid = {
    "estimator__n_estimators": [100, 200],
    "estimator__max_depth":    [5, 10, None],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

---

## 📦 Dependencies

```
scikit-learn >= 1.3.0
pandas       >= 2.0.0
numpy        >= 1.24.0
joblib       >= 1.3.0
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Dataset by **Stephan Matzka** — [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- Built with [scikit-learn](https://scikit-learn.org/)
