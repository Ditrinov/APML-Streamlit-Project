# predict.py  —  Inference script
# Run: python predict.py
# Loads the saved pipeline and predicts on a new sample.

import joblib
import pandas as pd

MODEL_PATH = "models/pdm_pipeline.joblib"


def load_pipeline(model_path: str):
    pipeline = joblib.load(model_path)
    print(f"[LOAD]  Pipeline loaded ← {model_path}")
    return pipeline


def predict_sample(pipeline, sample: dict) -> None:
    df = pd.DataFrame([sample])
    pred  = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0, 1]
    label = "⚠  FAILURE" if pred == 1 else "✓  NORMAL"
    print(f"\n[RESULT]  {label}  (failure probability: {proba:.4f})")


if __name__ == "__main__":
    pipeline = load_pipeline(MODEL_PATH)

    # Example new data point (raw format — same as training CSV)
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

    predict_sample(pipeline, new_sample)
