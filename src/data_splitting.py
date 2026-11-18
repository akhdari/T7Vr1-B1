"""Leakage-aware feature definition and reproducible holdout split."""

from sklearn.model_selection import train_test_split

from .data_cleaning import TARGET

# These fields describe the casualties or the assessed outcome and are not
# available as legitimate pre-event predictors for this project question.
POST_EVENT_COLUMNS = {
    "Accident_severity",
    "Casualty_class",
    "Casualty_gender",
    "Age_band_of_casualty",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Pedestrian_movement",
}


def split_features_target(data, test_size=0.2, random_state=42):
    """Return X_train, X_test, y_train, y_test after excluding post-event data."""
    if TARGET not in data:
        raise ValueError(f"Required target column {TARGET!r} is missing")
    excluded = POST_EVENT_COLUMNS | {TARGET}
    features = data.drop(columns=[c for c in excluded if c in data.columns])
    target = data[TARGET]
    return train_test_split(features, target, test_size=test_size, random_state=random_state)
