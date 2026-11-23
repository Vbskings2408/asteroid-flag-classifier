"""Predict the catalog flag from three measurements for one Earth encounter."""
import argparse
from math import isfinite
from pathlib import Path

import joblib
import pandas as pd

from dataset import FEATURES, MODEL


def predict(path, magnitude, speed, distance):
    if not all(isfinite(x) for x in (magnitude, speed, distance)) or speed < 0 or distance < 0:
        raise ValueError("Use finite numbers and a nonnegative speed and miss distance.")
    if not path.is_file():
        raise ValueError("Run train.py first to create the model.")
    # Load the model created by your own training run.
    saved = joblib.load(path)
    row = pd.DataFrame([[magnitude, speed, distance]], columns=FEATURES)
    score = saved["model"].predict_proba(row)[0, 1]
    label = "potentially hazardous" if score >= saved["cutoff"] else "unflagged"
    print(f"Predicted catalog label: {label}")
    print(f"Classifier score: {score:.4f} | Learned cutoff: {saved['cutoff']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magnitude", type=float, required=True, help="Absolute magnitude H")
    parser.add_argument("--speed", type=float, required=True, help="Relative speed in km/s")
    parser.add_argument("--distance", type=float, required=True, help="Encounter miss distance in au")
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    try:
        predict(args.model, args.magnitude, args.speed, args.distance)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
