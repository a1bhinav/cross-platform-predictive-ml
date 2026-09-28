import numpy as np
import onnxruntime as ort
import pytest
import torch

from export.common import parity
from export.export_onnx import export_model
from src.model import ConversionMLP, Predictor


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = Predictor(ConversionMLP(12), torch.arange(12).float(), torch.linspace(1, 20, 12)).eval()
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_model(model, path, [f"feature_{i}" for i in range(12)])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return model, session


@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_dynamic_batch_parity(exported, batch_size):
    model, session = exported
    x = np.random.default_rng(42).normal(5, 20, (batch_size, 12)).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(x)).numpy()
    actual = session.run(["probabilities"], {"features": x})[0]
    assert actual.shape == (batch_size, 1)
    assert parity(expected, actual)["max_absolute_error"] < 1e-5


def test_parity_rejects_mismatch_and_nan():
    with pytest.raises(AssertionError):
        parity(np.zeros(2), np.ones(2))
    with pytest.raises(AssertionError):
        parity(np.zeros(2), np.array([0.0, np.nan]))
