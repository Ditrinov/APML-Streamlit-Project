# train_pipeline.py  —  Entry point
# Run: python train_pipeline.py
# Dataset: AI4I 2020 Predictive Maintenance
# https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

from src.pipeline_builder import build_pipeline
from src.evaluate import evaluate_pipeline

# ── CONFIG ──────────────────────────────────────────────────
DATA_PATH    = "data/ai4i2020.csv"
MODEL_PATH   = "models/pdm_pipeline.joblib"
TARGET_COL   = "Machine failure"
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5
# ────────────────────────────────────────────────────────────


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[DATA]  Loaded  : {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def split_features_target(df: pd.DataFrame):
    # Drop all failure-mode sub-columns; keep only sensor + type features
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"]
    feature_cols = [c for c in df.columns if c not in failure_cols]
    X = df[feature_cols]
    y = df[TARGET_COL]
    print(f"[DATA]  Features: {feature_cols}")
    print(f"[DATA]  Target distribution:\n{y.value_counts().to_string()}")
    return X, y


def save_pipeline(pipeline, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"[SAVE]  Pipeline saved  → {output_path}")


def main():
    # 1. Load
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[SPLIT] Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

    # 3. Build pipeline
    pipeline = build_pipeline(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # 4. Train
    print("\n[TRAIN] Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    print("[TRAIN] Done.")

    # 5. Cross-validation
    print(f"\n[CV]    Running {CV_FOLDS}-Fold Cross Validation (ROC-AUC)...")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1,
    )
    print(f"[CV]    ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 6. Evaluate on test set
    evaluate_pipeline(pipeline, X_test, y_test)

    # 7. Save
    save_pipeline(pipeline, MODEL_PATH)

    print("\n[DONE]  Training complete!")


if __name__ == "__main__":
    main()
