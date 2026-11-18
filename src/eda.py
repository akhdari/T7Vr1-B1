"""Exploratory summaries and figures for Phase 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def missingness_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return missing count and percentage for each column, most missing first."""
    summary = pd.DataFrame({
        "missing_count": data.isna().sum(),
        "missing_percent": data.isna().mean().mul(100),
    })
    return summary.sort_values("missing_percent", ascending=False)


def save_phase1_figures(data: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save target distribution and top categorical frequency plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=data, x="Number_of_casualties", color="steelblue", ax=ax)
    ax.set(title="Casualties per accident", xlabel="Number of casualties", ylabel="Accidents")
    fig.tight_layout()
    path = output_dir / "casualties_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    for column in ["Accident_severity", "Weather_conditions", "Road_surface_conditions"]:
        if column not in data.columns:
            continue
        counts = data[column].fillna("Missing").value_counts().head(12).sort_values()
        fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(counts))))
        counts.plot.barh(ax=ax, color="slateblue")
        ax.set(title=f"Accidents by {column}", xlabel="Accidents", ylabel=column)
        fig.tight_layout()
        path = output_dir / f"{column.lower()}_counts.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
