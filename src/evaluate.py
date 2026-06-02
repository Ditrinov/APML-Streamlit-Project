# src/evaluate.py
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Print and return evaluation metrics for a trained pipeline.

    Returns
    -------
    dict with keys: roc_auc, confusion_matrix, report
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc   = roc_auc_score(y_test, y_proba)
    cm    = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["No Failure", "Failure"]
    )

    sep = "=" * 52
    print(f"\n{sep}")
    print("  EVALUATION REPORT")
    print(sep)
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print(f"\nROC-AUC Score : {roc:.4f}")
    print(sep)

    return {"roc_auc": roc, "confusion_matrix": cm, "report": report}
