# src/cleaner.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Step 1 — Data Cleaning:
    - Drop irrelevant ID columns (UDI, Product ID)
    - Encode categorical column 'Type' (L/M/H -> 0/1/2)
    - Rename columns: strip units, lowercase, replace spaces with underscore
    - Fill missing numeric values with median
    """

    def __init__(self, drop_cols=None, encode_col="Type"):
        self.drop_cols = drop_cols or ["UDI", "Product ID"]
        self.encode_col = encode_col
        self.label_encoder_ = LabelEncoder()

    def fit(self, X, y=None):
        if self.encode_col in X.columns:
            self.label_encoder_.fit(X[self.encode_col])
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Drop irrelevant columns
        cols_to_drop = [c for c in self.drop_cols if c in X.columns]
        X.drop(columns=cols_to_drop, inplace=True)

        # Encode categorical column
        if self.encode_col in X.columns:
            X[self.encode_col] = self.label_encoder_.transform(X[self.encode_col])

        # Rename columns: remove units in brackets, lowercase, underscore
        X.columns = (
            X.columns
             .str.strip()
             .str.replace(r"\s*\[.*?\]", "", regex=True)
             .str.replace(" ", "_")
             .str.lower()
        )

        # Fill missing numeric values with median
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col].fillna(X[col].median(), inplace=True)

        return X
