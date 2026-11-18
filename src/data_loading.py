"""Load the project accident dataset from a path independent of the cwd."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "accident_data.csv"


def load_accident_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the source CSV while treating common empty markers as missing."""
    data = pd.read_csv(path, na_values=["na", "NA", "N/A", ""], keep_default_na=True)
    data.columns = data.columns.str.strip()
    return data
