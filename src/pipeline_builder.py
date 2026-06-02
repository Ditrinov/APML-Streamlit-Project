# src/pipeline_builder.py
from sklearn.pipeline import Pipeline
from src.cleaner import DataCleaner
from src.scaler import DataScaler
from src.estimator import PDMEstimator


def build_pipeline(
    n_estimators: int = 100,
    max_depth: int = 10,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Pipeline:
    """
    Assemble and return the full sklearn Pipeline:
        DataCleaner -> DataScaler -> PDMEstimator

    Parameters
    ----------
    n_estimators  : Number of trees in RandomForest
    max_depth     : Max depth of each tree
    class_weight  : Weight strategy for imbalanced classes
    random_state  : Reproducibility seed

    Returns
    -------
    sklearn.pipeline.Pipeline (untrained)
    """
    pipeline = Pipeline(
        steps=[
            ("cleaner",   DataCleaner()),
            ("scaler",    DataScaler()),
            ("estimator", PDMEstimator(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=random_state,
            )),
        ]
    )
    return pipeline
