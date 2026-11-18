"""Trainable preprocessing that keeps imputation inside the CV pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical imputing/encoding pipelines."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop")
