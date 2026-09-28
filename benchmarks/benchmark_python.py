"""CPU runtime adapters and synchronous warm inference timing."""

import argparse
import json
import platform
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from export.common import check_export
from src.model import load_predictor

Predict = Callable[[np.ndarray], np.ndarray]


def load_runtime(runtime: str, artifacts: Path, batch_size: int) -> Predict:
    if runtime == "pytorch":
        torch.set_num_threads(1)
        model, _ = load_predictor(artifacts)

        @torch.inference_mode()
        def predict(x: np.ndarray) -> np.ndarray:
            return model(torch.from_numpy(x)).numpy().reshape(-1)

        return predict
    metadata = check_export(artifacts, runtime)
    if runtime == "onnx":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(artifacts / "model.onnx"), options,
                                       providers=["CPUExecutionProvider"])
        return lambda x: session.run(["probabilities"], {"features": x})[0].reshape(-1)
    if runtime == "coreml":
        if platform.system() != "Darwin":
            raise RuntimeError("Core ML inference requires macOS.")
        if batch_size > metadata["batch_max"]:
            raise ValueError("Core ML supports batches up to 4096; reduce --batch-size.")
        import coremltools as ct
        model = ct.models.MLModel(str(artifacts / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
        return lambda x: np.asarray(model.predict({"features": x})["probabilities"]).reshape(-1)
    if runtime == "litert":
        if batch_size != metadata["batch_size"]:
            raise ValueError("LiteRT uses a fixed batch; re-export with the requested --batch-size.")
        from ai_edge_litert.compiled_model import CompiledModel
        model = CompiledModel.from_file(str(artifacts / "model.tflite"))
        inputs = model.create_input_buffers(0)
        outputs = model.create_output_buffers(0)

        def predict(x: np.ndarray) -> np.ndarray:
            inputs[0].write(np.ascontiguousarray(x, dtype=np.float32))
            model.run_by_index(0, inputs, outputs)
            return outputs[0].read(batch_size, np.float32).copy().reshape(-1)

        return predict
    raise ValueError(f"Unknown runtime: {runtime}")


def predict_all(predict: Predict, x: np.ndarray, batch_size: int) -> np.ndarray:
    values = []
    for offset in range(0, len(x), batch_size):
        batch = x[offset:offset + batch_size]
        size = len(batch)
        if size < batch_size:
            batch = np.pad(batch, ((0, batch_size - size), (0, 0)))
        values.append(predict(batch)[:size])
    return np.concatenate(values)


def measure(predict: Predict, batch: np.ndarray, iterations: int, warmup: int) -> dict:
    if iterations < 1 or warmup < 0 or not len(batch):
        raise ValueError("Need positive iterations/batch and nonnegative warmup.")
    for _ in range(warmup):
        predict(batch)
    elapsed = []
    for _ in range(iterations):
        start = perf_counter_ns()
        predict(batch)
        elapsed.append((perf_counter_ns() - start) / 1e6)
    return {"mean_ms": float(np.mean(elapsed)), "p50_ms": float(np.percentile(elapsed, 50)),
            "p95_ms": float(np.percentile(elapsed, 95)),
            "samples_per_sec": float(len(batch) * 1000 / np.mean(elapsed))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["pytorch", "onnx", "coreml", "litert"], default="onnx")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    with np.load(args.artifacts / "test.npz") as data:
        batch = data["x"][:args.batch_size]
    if args.batch_size < 1 or len(batch) != args.batch_size:
        parser.error("batch-size must be positive and no larger than the test set")
    predict = load_runtime(args.runtime, args.artifacts, args.batch_size)
    print(json.dumps({"runtime": args.runtime, "batch_size": len(batch),
                      **measure(predict, batch, args.iterations, args.warmup)}, indent=2))


if __name__ == "__main__":
    main()
