# src/scaler.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class DataScaler(BaseEstimator, TransformerMixin):
    """
    Step 2 — Feature Scaling:
    - Apply StandardScaler only to continuous numeric sensor columns
    - Categorical encoded column (type) is left unchanged
    """

    def __init__(self, scale_cols=None):
        self.scale_cols = scale_cols or [
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
        ]
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        cols = [c for c in self.scale_cols if c in X.columns]
        self.scaler_.fit(X[cols])
        self.fitted_cols_ = cols
        return self

    def transform(self, X, y=None):
        X = X.copy()
        cols = [c for c in self.fitted_cols_ if c in X.columns]
        X[cols] = self.scaler_.transform(X[cols])
        return X
