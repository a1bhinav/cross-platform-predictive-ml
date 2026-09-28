"""Convert with Google's litert-torch and execute with LiteRT CompiledModel."""

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
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if platform.system() != "Linux":
        print("LiteRT conversion not run: litert-torch documents Linux support. "
              "Use Python 3.11 on Linux, install with pip install -e '.[litert]', "
              "copy artifacts/ there, and rerun python -m export.export_litert.")
        return 2
    try:
        import litert_torch
    except (ImportError, OSError) as error:
        print(f"LiteRT converter unavailable: {error}. Install: pip install -e '.[litert]'")
        return 2
    torch.set_num_threads(1)
    model, metadata = load_predictor(args.artifacts)
    example = torch.zeros(args.batch_size, len(metadata["feature_names"]))
    path = args.artifacts / "model.tflite"
    try:
        converted = litert_torch.convert(model.eval(), (example,))
        converted.export(str(path))
    except Exception as error:
        print(f"LiteRT conversion failed: {error}. Check the supported Linux toolchain in README.")
        return 1
    record_export(args.artifacts, "litert", batch_size=args.batch_size,
                  conversion="completed", prediction="not_run")
    print(f"Saved {path} (conversion completed).")
    try:
        from benchmarks.benchmark_python import load_runtime
        predict = load_runtime("litert", args.artifacts, args.batch_size)
        with np.load(args.artifacts / "test.npz") as data:
            x = data["x"][:args.batch_size]
        if len(x) < args.batch_size:
            x = np.pad(x, ((0, args.batch_size - len(x)), (0, 0)))
        actual = predict(x).reshape(-1, 1)
    except Exception as error:
        print(f"Conversion succeeded, but LiteRT execution is unavailable: {error}")
        return 2
    with torch.inference_mode():
        result = parity(model(torch.from_numpy(x)).numpy(), actual)
    record_export(args.artifacts, "litert", batch_size=args.batch_size,
                  conversion="completed", prediction="passed", parity=result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
