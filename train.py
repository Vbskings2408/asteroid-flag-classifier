"""Train a shallow decision tree and measure recall on unseen asteroid IDs."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from dataset import FEATURES, MODEL, SAMPLE, one_per_asteroid, output_path

SEED = 42


def split_objects(data):
    counts = data["pha"].value_counts()
    if min(counts.get(0, 0), counts.get(1, 0)) < 10:
        raise ValueError("Need at least 10 different asteroids of each class. Fetch a longer period.")
    train_val, test = train_test_split(
        data, test_size=0.20, stratify=data["pha"], random_state=SEED
    )
    # One quarter of the remaining 80% makes a 20% validation set.
    train, validation = train_test_split(
        train_val, test_size=0.25, stratify=train_val["pha"], random_state=SEED
    )
    return train, validation, test


def choose_threshold(actual, scores, min_recall):
    """Among cutoffs meeting the validation recall goal, prefer higher precision."""
    precision, recall, cutoffs = precision_recall_curve(actual, scores)
    # The final precision/recall pair has no corresponding cutoff.
    eligible = np.flatnonzero(recall[:-1] >= min_recall)
    best_index = max(eligible, key=lambda i: (precision[i], recall[i], cutoffs[i]))
    return float(cutoffs[best_index])


def summarize(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Recall": tp / (tp + fn) if tp + fn else 0.0,
        "Precision": tp / (tp + fp) if tp + fp else 0.0,
    }


def train(csv_path, output, min_recall):
    if not 0 < min_recall <= 1:
        raise ValueError("Choose a recall goal greater than 0 and at most 1.")
    if output.resolve() == csv_path.resolve():
        raise ValueError("Give the model a different path from the input CSV.")
    data = one_per_asteroid(pd.read_csv(csv_path, dtype={"asteroid_id": "string"}))
    fit, val, test = split_objects(data)
    model = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=SEED
    )
    model.fit(fit[FEATURES], fit["pha"])

    # Pick the cutoff with validation labels only. Test labels are used below for reporting.
    val_scores = model.predict_proba(val[FEATURES])[:, 1]
    cutoff = choose_threshold(val["pha"], val_scores, min_recall)
    test_scores = model.predict_proba(test[FEATURES])[:, 1]
    result = summarize(test["pha"], test_scores >= cutoff)
    baselines = {
        "Tree": result,
        "Brightness only (H <= 22)": summarize(test["pha"], test["absolute_magnitude"] <= 22),
        "Flag every asteroid": summarize(test["pha"], np.ones(len(test), dtype=int)),
    }

    with output_path(output) as temporary:
        joblib.dump({"model": model, "cutoff": cutoff}, temporary)
    print(f"Unique objects: {len(data)} | Flagged: {int(data['pha'].sum())}")
    print(f"Training: {len(fit)} | Validation: {len(val)} | Test: {len(test)}")
    val_result = summarize(val["pha"], val_scores >= cutoff)
    print(f"Validation recall goal: {min_recall:.1%} | Achieved: {val_result['Recall']:.1%} | Cutoff: {cutoff:.4f}")
    print("\nHeld-out test results")
    print(pd.DataFrame.from_dict(baselines, orient="index").to_string(
        formatters={"Recall": "{:.1%}".format, "Precision": "{:.1%}".format}
    ))
    print(f"\nSaved model to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SAMPLE)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--min-recall", type=float, default=0.90)
    args = parser.parse_args()
    try:
        train(args.data, args.model, args.min_recall)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
