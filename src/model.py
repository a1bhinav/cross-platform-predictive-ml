"""A logits MLP and the shared raw-feature inference contract."""

from pathlib import Path

import torch
from torch import nn


class ConversionMLP(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return [batch, 1] logits from standardized features."""
        return self.network(features)


class Predictor(nn.Module):
    """Every exported model includes training-only scaling and sigmoid."""

    def __init__(self, model: ConversionMLP, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean.to(torch.float32))
        self.register_buffer("scale", scale.to(torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((features - self.mean) / self.scale))


def load_predictor(artifacts: Path) -> tuple[Predictor, dict]:
    checkpoint = torch.load(artifacts / "model.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    model = ConversionMLP(len(metadata["feature_names"]))
    model.load_state_dict(checkpoint["model_state"])
    predictor = Predictor(model, torch.tensor(metadata["scaler_mean"]),
                          torch.tensor(metadata["scaler_scale"])).eval()
    return predictor, metadata
