"""Direct TorchScript -> Core ML ML Program conversion with float32 compute."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from export.common import parity, record_export
from src.model import load_predictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        import coremltools as ct
    except (ImportError, OSError) as error:
        print(f"Core ML unavailable: {error}. Install with: pip install -e '.[coreml]'")
        return 2
    torch.set_num_threads(1)
    model, metadata = load_predictor(args.artifacts)
    example = torch.zeros(1, len(metadata["feature_names"]))
    try:
        traced = torch.jit.trace(model, example)
        converted = ct.convert(
            traced, convert_to="mlprogram",
            inputs=[ct.TensorType(name="features", dtype=np.float32,
                                 shape=(ct.RangeDim(1, 4096, default=1), example.shape[1]))],
            outputs=[ct.TensorType(name="probabilities", dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        converted.save(str(args.artifacts / "model.mlpackage"))
    except Exception as error:
        print(f"Core ML conversion failed: {error}. Check coremltools/PyTorch compatibility in README.")
        return 1
    record_export(args.artifacts, "coreml", batch_max=4096, conversion="completed",
                  prediction="not_run", coremltools_version=ct.__version__)
    print("Saved artifacts/model.mlpackage (conversion completed).")
    if platform.system() != "Darwin":
        print("Prediction not run: Core ML execution requires macOS 13 or newer.")
        return 0
    with np.load(args.artifacts / "test.npz") as data:
        x = data["x"][:256]
    try:
        actual = np.asarray(converted.predict({"features": x})["probabilities"])
    except Exception as error:
        print(f"Conversion succeeded, but Core ML prediction is unavailable: {error}")
        return 2
    with torch.inference_mode():
        result = parity(model(torch.from_numpy(x)).numpy(), actual)
    record_export(args.artifacts, "coreml", batch_max=4096, conversion="completed",
                  prediction="passed", parity=result, samples=len(x), coremltools_version=ct.__version__)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
