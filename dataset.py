"""Shared schema: keep one Earth encounter per asteroid before splitting the data."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "historical.csv"
MODEL = ROOT / "model.joblib"
FEATURES = ["absolute_magnitude", "speed_km_s", "miss_distance_au"]
COLUMNS = ["asteroid_id", "approach_date", *FEATURES, "pha"]


@contextmanager
def output_path(destination):
    """Write to a temporary file, then replace the destination after success."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def one_per_asteroid(frame):
    """Validate the six columns and keep each ID's earliest available encounter."""
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("No Earth encounters found in this dataset.")

    data = frame[COLUMNS].copy()
    data["asteroid_id"] = data["asteroid_id"].astype("string").str.strip()
    if data.isna().any().any() or data["asteroid_id"].eq("").any():
        raise ValueError("IDs, dates, labels and features must all be present.")

    dates = pd.to_datetime(data["approach_date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError("Each encounter needs a valid calendar date.")
    data["approach_date"] = dates.dt.strftime("%Y-%m-%d")
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data[FEATURES]).all().all():
        raise ValueError("Features must be finite numbers.")
    if (data[["speed_km_s", "miss_distance_au"]] < 0).any().any():
        raise ValueError("Speed and miss distance must be nonnegative.")
    labels = pd.to_numeric(data["pha"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Use 1 for a flagged object and 0 for an unflagged object.")
    data["pha"] = labels.astype(int)
    if (data.groupby("asteroid_id")["pha"].nunique() > 1).any():
        raise ValueError("An ID has conflicting flags. Use one catalog snapshot.")

    # Break same-day ties consistently. IDs and dates never enter the model.
    order = ["asteroid_id", "approach_date", "miss_distance_au", "speed_km_s", "absolute_magnitude"]
    return data.sort_values(order).drop_duplicates("asteroid_id").reset_index(drop=True)
