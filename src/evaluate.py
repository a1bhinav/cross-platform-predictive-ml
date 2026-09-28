"""Evaluate held-out predictions, including equal-width calibration bins."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.model import load_predictor


def metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    p = probabilities.reshape(-1)
    if len(p) != len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected one finite probability in [0, 1] per label.")
    predicted = p >= 0.5
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        if mask.any():
            bins.append({"lower": i / 10, "upper": (i + 1) / 10, "count": int(mask.sum()),
                         "mean_probability": float(p[mask].mean()),
                         "observed_rate": float(y[mask].mean())})
    ece = sum(b["count"] * abs(b["mean_probability"] - b["observed_rate"]) for b in bins) / len(y)
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc_average_precision": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "brier_score": float(brier_score_loss(y, p)), "threshold": 0.5,
            "positive_rate": float(y.mean()), "ece_10_bins": ece, "calibration_bins": bins}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, _ = load_predictor(args.artifacts)
    with np.load(args.artifacts / "test.npz") as data, torch.inference_mode():
        p = np.concatenate([model(batch).numpy().ravel()
                            for batch in torch.from_numpy(data["x"]).split(4096)])
        result = metrics(data["y"], p)
    (args.artifacts / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
