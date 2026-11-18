"""Basic, auditable cleaning for the accident severity dataset."""

import pandas as pd


TARGET = "Number_of_casualties"


def clean_accident_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing markers and types without dropping incomplete rows.

    Missing feature values are retained for imputation inside model pipelines.
    Rows with an absent or non-positive target are excluded because they cannot
    represent a valid casualty-count regression observation.
    """
    cleaned = data.copy()
    cleaned.columns = cleaned.columns.str.strip()
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
    cleaned = cleaned.replace({"na": pd.NA, "NA": pd.NA, "N/A": pd.NA})
    if TARGET not in cleaned.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing")
    cleaned[TARGET] = pd.to_numeric(cleaned[TARGET], errors="coerce")
    cleaned = cleaned.dropna(subset=[TARGET])
    cleaned = cleaned.loc[cleaned[TARGET] > 0].copy()
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned
