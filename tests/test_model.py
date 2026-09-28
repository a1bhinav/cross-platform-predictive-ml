import numpy as np
import pytest
import torch

from src.model import ConversionMLP, Predictor


@pytest.mark.parametrize("batch_size", [1, 7, 32])
def test_forward(batch_size):
    torch.manual_seed(42)
    model = ConversionMLP(12)
    logits = model(torch.randn(batch_size, 12))
    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_predictor_embeds_scaler():
    torch.manual_seed(42)
    model = ConversionMLP(12).eval()
    mean, scale = torch.arange(12, dtype=torch.float32), torch.full((12,), 2.0)
    x = torch.randn(9, 12)
    predictor = Predictor(model, mean, scale).eval()
    with torch.inference_mode():
        actual = predictor(x)
        expected = torch.sigmoid(model((x - mean) / scale))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert ((actual >= 0) & (actual <= 1)).all()
