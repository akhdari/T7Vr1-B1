# Group 1 - Theme 7: Road Accident Regression

This project uses the Addis Ababa City Road Traffic Accident Severity Dataset to
explore and model the number of casualties per accident.

## Phase 1: Steps 0-3

Run `python phase1_steps0_3.py` from any working directory. It uses the local
CSV when available and downloads the specified Figshare file if needed. The
script profiles data quality and reporting limitations, cleans the target and
duplicates, excludes casualty/outcome leakage fields from model inputs and the
export, prepares both One-Hot and Ordinal ElasticNet pipelines, and saves
`dataset_clean.csv` with a `split_ref` column. The reference labels mark the
first fold of the repeated 5-fold split; use the full 15-fold splitter during
model comparison. The notebook launches this same script. Model fitting and
performance evaluation are the next deliverable.

## Project layout

- `data/accident_data.csv` - unchanged source dataset
- `dataset_clean.csv` - cleaned modeling table with first-fold `split_ref` labels
- `phase1_steps0_3.py` - executable Steps 0-3 implementation
- `notebooks/T7Vr1-B1.ipynb` - notebook entry point for Steps 0-3
- `outputs/figures/` - generated target-distribution and correlation figures

## Setup and run

From the project directory, install the packages listed in `requirements.txt`,
then open and run `notebooks/T7Vr1-B1.ipynb` from top to bottom. Figures are
written to `outputs/figures/`.

Dataset source: [Addis Ababa City Road Traffic Accident Severity Dataset](https://doi.org/10.6084/m9.figshare.28122899.v1),
licensed under CC BY 4.0.
