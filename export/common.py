"""Small shared helpers for export provenance and numerical checks."""

import hashlib
import json
from pathlib import Path

import numpy as np


def checkpoint_hash(artifacts: Path) -> str:
    return hashlib.sha256((artifacts / "model.pt").read_bytes()).hexdigest()


def record_export(artifacts: Path, runtime: str, **metadata: object) -> None:
    record = {"checkpoint_sha256": checkpoint_hash(artifacts), **metadata}
    (artifacts / f"{runtime}_export.json").write_text(json.dumps(record, indent=2) + "\n")


def check_export(artifacts: Path, runtime: str) -> dict:
    metadata = json.loads((artifacts / f"{runtime}_export.json").read_text())
    if metadata["checkpoint_sha256"] != checkpoint_hash(artifacts):
        raise ValueError(f"Stale {runtime} export; re-export the current checkpoint.")
    return metadata


def parity(reference: np.ndarray, actual: np.ndarray, tolerance: float = 1e-5) -> dict:
    if reference.shape != actual.shape or not np.isfinite(actual).all():
        raise AssertionError("Runtime returned the wrong shape or non-finite predictions.")
    errors = np.abs(reference - actual)
    result = {"max_absolute_error": float(errors.max()), "mean_absolute_error": float(errors.mean())}
    if result["max_absolute_error"] >= tolerance:
        raise AssertionError(f"Prediction parity failed (tolerance {tolerance}): {result}")
    return result
