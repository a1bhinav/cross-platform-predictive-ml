"""Compare actual predictions/timings; retain blank metrics for unavailable runtimes."""

import argparse
import csv
import json
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from benchmarks.benchmark_python import load_runtime, measure, predict_all
from export.common import check_export, checkpoint_hash, parity
from src.model import load_predictor


def run_cpp(executable: Path, artifacts: Path, x: np.ndarray, names: list[str],
            batch_size: int, iterations: int, warmup: int) -> tuple[np.ndarray, dict]:
    check_export(artifacts, "onnx")
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        np.savetxt(folder / "input.csv", x, delimiter=",", fmt="%.9g", header=",".join(names), comments="")
        subprocess.run([str(executable.resolve()), "--model", str((artifacts / "model.onnx").resolve()),
                        "--input", str(folder / "input.csv"), "--output", str(folder / "predictions.csv"),
                        "--metrics", str(folder / "metrics.json"), "--batch-size", str(batch_size),
                        "--iterations", str(iterations), "--warmup", str(warmup)],
                       check=True, capture_output=True, text=True)
        return (np.loadtxt(folder / "predictions.csv", skiprows=1, ndmin=1),
                json.loads((folder / "metrics.json").read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"))
    parser.add_argument("--cpp-executable", type=Path, default=Path("build/predict"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    with np.load(args.artifacts / "test.npz") as data:
        x, y = data["x"], data["y"]
    if not 1 <= args.batch_size <= len(x) or args.iterations < 1 or args.warmup < 0:
        parser.error("Use 1 <= batch-size <= test rows, positive iterations, and nonnegative warmup.")
    reference = predict_all(load_runtime("pytorch", args.artifacts, args.batch_size), x, args.batch_size)
    _, metadata = load_predictor(args.artifacts)
    rows = []
    runtime_versions = {}
    failures = False
    for runtime, name in [("pytorch", "PyTorch"), ("onnx", "ONNX Runtime"), ("cpp", "ONNX Runtime C++"),
                          ("coreml", "Core ML"), ("litert", "LiteRT")]:
        row = {"runtime": name, "status": "unavailable", "batch_size": args.batch_size}
        try:
            if runtime == "cpp":
                actual, timing = run_cpp(args.cpp_executable, args.artifacts, x, metadata["feature_names"],
                                         args.batch_size, args.iterations, args.warmup)
                runtime_versions[name] = timing["onnxruntime_version"]
            else:
                predict = load_runtime(runtime, args.artifacts, args.batch_size)
                actual = predict_all(predict, x, args.batch_size)
                timing = measure(predict, x[:args.batch_size], args.iterations, args.warmup)
        except (ImportError, FileNotFoundError) as error:
            row["note"] = str(error)
            if runtime in {"pytorch", "onnx"}:
                failures = True
        except Exception as error:
            row.update(status="failed", note=str(error))
            failures = True
        else:
            try:
                errors = parity(reference, actual)
            except AssertionError as error:
                row.update(status="failed", note=str(error))
                failures = True
            else:
                row.update(status="measured", roc_auc=float(roc_auc_score(y, actual)),
                           max_error=errors["max_absolute_error"],
                           mean_error=errors["mean_absolute_error"],
                           **{key: timing[key] for key in ("mean_ms", "p50_ms", "p95_ms", "samples_per_sec")})
        rows.append(row)
        print(json.dumps(row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["runtime", "status", "roc_auc", "max_error", "mean_error", "mean_ms", "p50_ms", "p95_ms",
              "samples_per_sec", "batch_size", "note"]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    versions = {}
    for package in ("torch", "numpy", "onnxruntime", "coremltools", "litert-torch", "ai-edge-litert"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    context = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "os": platform.system(),
               "os_release": platform.release(), "architecture": platform.machine(),
               "python": platform.python_version(), "packages": versions,
               "native_runtime_versions": runtime_versions,
               "checkpoint_sha256": checkpoint_hash(args.artifacts), "test_rows": len(x),
               "iterations": args.iterations, "warmup": args.warmup, "batch_size": args.batch_size,
               "threads": "PyTorch/ORT/C++: 1; Core ML/LiteRT: runtime defaults", "device": "CPU"}
    args.output.with_suffix(".json").write_text(json.dumps(context, indent=2) + "\n")
    return int(failures)


if __name__ == "__main__":
    raise SystemExit(main())
