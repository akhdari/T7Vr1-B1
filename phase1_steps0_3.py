"""Deliverable 1: Steps 0-3 for Group 1, Theme 7.

Estimate the number of casualties per Addis Ababa road accident from recorded
accident circumstances. The script reads ``data/accident_data.csv`` relative to
this file, downloads the specified Figshare CSV if it is missing, and writes
``dataset_clean.csv`` and EDA figures beside this project.

Run from any working directory with::

    python path/to/T7Vr1-B1/phase1_steps0_3.py

This deliverable prepares two comparable ElasticNet pipelines. It does not fit
models or report predictive performance; those belong to the modeling step.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Literal

import matplotlib

# This script writes figures to disk and does not open interactive windows.
# Agg works in terminal, CI, remote, and headless environments without Tk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Step 0: setup and project configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_CSV = DATA_DIR / "accident_data.csv"
CLEAN_CSV = PROJECT_ROOT / "dataset_clean.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TARGET = "Number_of_casualties"

# Normalize these source tokens to missing values. Do not treat "Other" as
# unknown: it is a legitimate value in several source columns.
UNKNOWN_MARKERS = {"", "na", "n/a", "nan", "null", "none", "unknown", "unspecified"}

# The count limit is deliberately small. We report summaries and a few example
# levels rather than dumping every rare time-of-day value to the console.
RARE_LEVEL_MAX_COUNT = 10
RARE_EXAMPLES_PER_COLUMN = 5

# Outcome and casualty-assessment data must not be used to estimate the number
# of casualties from accident circumstances. Keep the target itself.
LEAKAGE_COLUMNS = {
    "Accident_severity",
    "Casualty_class",
    "Casualty_gender",
    "Age_band_of_casualty",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Pedestrian_movement",
}


def download_source_csv(destination: Path = SOURCE_CSV) -> Path:
    """Download the required CSV from Figshare article 28122899.

    The Figshare API is used to resolve the direct download URL for file
    51446597. Downloading to a temporary file prevents a partial transfer from
    being mistaken for a complete local dataset on a later run.
    """
    api_url = "https://api.figshare.com/v2/articles/28122899"
    with urllib.request.urlopen(api_url, timeout=30) as response:
        article = json.load(response)

    file_record = next(
        (item for item in article.get("files", []) if int(item["id"]) == 51446597),
        None,
    )
    if file_record is None:
        raise RuntimeError("Figshare article does not contain expected file 51446597")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    try:
        urllib.request.urlretrieve(file_record["download_url"], temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def load_source_data(path: Path = SOURCE_CSV) -> pd.DataFrame:
    """Read the source CSV and normalize whitespace and unknown markers."""
    if not path.exists():
        print(f"Dataset not found at {path}; downloading from Figshare.")
        download_source_csv(path)

    data = pd.read_csv(path, na_values=["na", "NA", "N/A", ""], keep_default_na=True)
    data.columns = data.columns.str.strip()

    # StringDtype preserves missing values as <NA> while normalizing tokens.
    for column in data.select_dtypes(include=["object", "string"]).columns:
        values = data[column].astype("string").str.strip()
        data[column] = values.mask(values.str.casefold().isin(UNKNOWN_MARKERS), pd.NA)
    return data


# ---------------------------------------------------------------------------
# Step 1: exploration and data-quality review
# ---------------------------------------------------------------------------
def print_data_quality_report(data: pd.DataFrame) -> None:
    """Print dataset shape, types, missingness, duplicates, and rare levels."""
    print("\n[Step 1] Source data overview")
    print(f"Shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print("\nColumn data types:\n", data.dtypes.to_string())
    print(f"\nExact duplicate rows before cleaning: {int(data.duplicated().sum()):,}")

    missing_report = pd.DataFrame(
        {
            "missing_count": data.isna().sum(),
            "missing_percent": (data.isna().mean() * 100).round(2),
        }
    ).sort_values(["missing_percent", "missing_count"], ascending=False)
    print("\nMissing/unknown values (top 20 columns):\n", missing_report.head(20).to_string())

    rare_rows = []
    categorical = data.select_dtypes(include=["object", "string", "category"])
    for column in categorical.columns:
        counts = data[column].value_counts(dropna=True)
        rare = counts[counts <= RARE_LEVEL_MAX_COUNT].sort_values()
        if rare.empty:
            continue
        examples = [f"{level!r} ({int(count)})" for level, count in rare.head(RARE_EXAMPLES_PER_COLUMN).items()]
        rare_rows.append(
            {
                "column": column,
                "rare_levels": int(rare.size),
                "rows_in_rare_levels": int(rare.sum()),
                "examples (up to 5)": "; ".join(examples),
            }
        )
    print(f"\nRare categories (level count <= {RARE_LEVEL_MAX_COUNT}; summarized):")
    if rare_rows:
        print(pd.DataFrame(rare_rows).to_string(index=False))
    else:
        print("No rare categories at this threshold.")


def analyze_target_and_correlations(data: pd.DataFrame) -> pd.Series:
    """Report target distribution/skew and numeric feature correlations."""
    if TARGET not in data.columns:
        raise KeyError(f"Expected target column {TARGET!r}; found {data.columns.tolist()}")
    target = pd.to_numeric(data[TARGET], errors="coerce")
    print("\nTarget summary:\n", target.describe().to_string())
    print(f"Target skewness: {target.skew():.4f}")
    print("Target value counts:\n", target.value_counts(dropna=False).sort_index().to_string())

    numeric_data = data.select_dtypes(include=np.number).copy()
    numeric_data[TARGET] = target
    correlations = (
        numeric_data.corr(numeric_only=True)[TARGET]
        .drop(labels=[TARGET], errors="ignore")
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    print("\nNumeric feature correlations with target (descriptive only):\n", correlations.to_string())
    return target


def save_eda_figures(data: pd.DataFrame, target: pd.Series) -> None:
    """Save a target distribution plot and a numeric correlation heatmap."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(target.dropna(), discrete=True, ax=ax)
    ax.set(title="Number of casualties per accident", xlabel=TARGET, ylabel="Accident records")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)

    numeric_data = data.select_dtypes(include=np.number).copy()
    numeric_data[TARGET] = target
    corr = numeric_data.corr(numeric_only=True)
    if len(corr.columns) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        ax.set_title("Numeric feature correlations (descriptive)")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "numeric_correlations.png", dpi=150)
        plt.close(fig)


def print_leakage_and_reporting_audit(data: pd.DataFrame) -> None:
    """Document known leakage risks and limits of the observed records."""
    present = sorted(LEAKAGE_COLUMNS.intersection(data.columns))
    print("\nLeakage audit: outcome/casualty columns excluded from predictors:", present)
    print(
        "Review remaining fields for when they become known; exclude any field "
        "recorded only after the accident or casualty assessment."
    )
    print(
        "Reporting-bias audit: the data reflects reported and documented "
        "accidents. Missingness may be non-random, rare levels may reflect "
        "limited documentation, and this dataset cannot quantify unreported "
        "accidents or establish representativeness by itself."
    )


# ---------------------------------------------------------------------------
# Step 2: cleaning, leakage removal, encoders, and ElasticNet pipelines
# ---------------------------------------------------------------------------
def clean_data(data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Validate the numeric positive target and remove exact duplicate rows."""
    cleaned = data.copy()
    cleaned[TARGET] = target
    cleaned = cleaned.dropna(subset=[TARGET])
    cleaned = cleaned.loc[cleaned[TARGET] > 0].copy()
    rows_before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    removed = rows_before_deduplication - len(cleaned)
    print(
        f"\n[Step 2] Removed {removed:,} duplicate rows; retained "
        f"{len(cleaned):,} rows with a positive numeric target."
    )
    return cleaned


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a sparse OneHotEncoder to control memory for high-cardinality data."""
    return OneHotEncoder(handle_unknown="ignore")


def build_elasticnet_pipeline(
    encoding: Literal["onehot", "ordinal"],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Build an imputation -> encoding -> scaling -> ElasticNet pipeline.

    All learned preprocessing stays inside the pipeline, so later cross
    validation fits imputers, encoders, and scaling using each training fold.
    """
    if encoding == "onehot":
        encoder = make_one_hot_encoder()
    elif encoding == "ordinal":
        # This is a comparison option. Ordinal codes impose an arbitrary order
        # on nominal labels, so they should not be interpreted as ranks.
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError("encoding must be 'onehot' or 'ordinal'")

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            # Centering a sparse one-hot matrix would make it dense. Scaling
            # without centering preserves sparsity and still standardizes it.
            ("scaler", StandardScaler(with_mean=False)),
            ("model", ElasticNet(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def prepare_features_and_pipelines(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, Pipeline, Pipeline]:
    """Remove leakage fields from predictors and return both encoder options."""
    excluded = LEAKAGE_COLUMNS | {TARGET, "split_ref"}
    features = cleaned.drop(columns=[column for column in excluded if column in cleaned.columns])
    target = cleaned[TARGET].astype(float)
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    print("Excluded leakage columns:", sorted(LEAKAGE_COLUMNS.intersection(cleaned.columns)))
    print(
        f"Candidate predictors: {len(features.columns)} "
        f"(numeric: {len(numeric_columns)}; categorical: {len(categorical_columns)})"
    )
    onehot_pipeline = build_elasticnet_pipeline("onehot", numeric_columns, categorical_columns)
    ordinal_pipeline = build_elasticnet_pipeline("ordinal", numeric_columns, categorical_columns)
    print("Prepared One-Hot and Ordinal ElasticNet pipelines; model fitting is the next deliverable.")
    return features, target, onehot_pipeline, ordinal_pipeline


# ---------------------------------------------------------------------------
# Step 3: repeated K-fold reference split and cleaned CSV output
# ---------------------------------------------------------------------------
def add_reference_split(cleaned: pd.DataFrame, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Add train/test labels for the first fold of the repeated 5x3 CV split.

    RepeatedKFold shuffles each fold internally when a random_state is set; it
    has no separate ``shuffle`` argument. ``split_ref`` is a single reference
    fold for export and inspection. Model comparison must use all 15 splits.
    """
    reference_cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    _first_train_indices, first_test_indices = next(reference_cv.split(features, target))

    result = cleaned.copy()
    result["split_ref"] = "train"
    result.loc[result.index[first_test_indices], "split_ref"] = "test"

    # The exported modeling table contains the target and legitimate features,
    # but does not carry post-outcome leakage columns.
    export_columns = [column for column in result.columns if column not in LEAKAGE_COLUMNS]
    result = result.loc[:, export_columns]
    result.to_csv(CLEAN_CSV, index=False)
    print("\n[Step 3] Reference splitter: RepeatedKFold(5 folds x 3 repeats, random_state=42)")
    print("split_ref labels the first fold only; use all 15 folds for model comparison.")
    print(result["split_ref"].value_counts().to_string())
    print(f"Saved cleaned, leakage-filtered dataset: {CLEAN_CSV}")
    return result


def main() -> None:
    """Run Deliverable 1 in order and write the requested artifacts."""
    print(f"Group 1 / Theme 7 | scikit-learn {sklearn.__version__} | random_state={RANDOM_STATE}")
    source = load_source_data()
    print_data_quality_report(source)
    target = analyze_target_and_correlations(source)
    print_leakage_and_reporting_audit(source)
    save_eda_figures(source, target)

    cleaned = clean_data(source, target)
    features, y, _onehot_pipeline, _ordinal_pipeline = prepare_features_and_pipelines(cleaned)
    add_reference_split(cleaned, features, y)


if __name__ == "__main__":
    main()
