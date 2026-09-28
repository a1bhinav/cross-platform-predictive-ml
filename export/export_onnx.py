"""Export raw features -> probabilities with a dynamic ONNX batch dimension."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from export.common import checkpoint_hash, parity, record_export
from src.model import Predictor, load_predictor


def export_model(model: Predictor, path: Path, feature_names: list[str], model_id: str = "test") -> None:
    example = torch.zeros(2, len(feature_names), dtype=torch.float32)
    # Explicitly select the stable trace exporter; no onnxscript dependency needed.
    torch.onnx.export(model.eval(), example, str(path), input_names=["features"],
                      output_names=["probabilities"], opset_version=17, dynamo=False,
                      dynamic_axes={"features": {0: "batch"}, "probabilities": {0: "batch"}})
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(graph, {"feature_names": ",".join(feature_names),
                                       "checkpoint_sha256": model_id,
                                       "input_contract": "raw float32; scaler embedded"})
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, metadata = load_predictor(args.artifacts)
    path = args.artifacts / "model.onnx"
    export_model(model, path, metadata["feature_names"], checkpoint_hash(args.artifacts))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    with np.load(args.artifacts / "test.npz") as data, torch.inference_mode():
        x = data["x"]
        reference = np.concatenate([model(torch.from_numpy(b)).numpy() for b in np.array_split(x, max(1, len(x) // 1024))])
        actual = session.run(["probabilities"], {"features": x})[0]
        result = parity(reference, actual)
    record_export(args.artifacts, "onnx", parity=result, samples=len(x), batch="dynamic")
    print(json.dumps({"artifact": str(path), **result}, indent=2))


if __name__ == "__main__":
    main()
