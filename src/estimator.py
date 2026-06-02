# src/estimator.py
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


class PDMEstimator(BaseEstimator, ClassifierMixin):
    """
    Step 3 — Classification Estimator:
    - Wraps RandomForestClassifier for Predictive Maintenance
    - Uses class_weight='balanced' to handle class imbalance (~97% no-failure)
    - Drop-in replaceable with XGBoost / LightGBM if needed
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            scale_pos_weight=33,   # handles imbalance: ~97/3 ratio
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, y):
        return self.model_.score(X, y)
