"""Fit a small, deterministic CPU model; select the epoch by validation loss."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import ConversionMLP


def train(artifacts: Path, epochs: int, batch_size: int, seed: int) -> None:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata = json.loads((artifacts / "data_metadata.json").read_text())
    with np.load(artifacts / "train.npz") as data:
        scaler = StandardScaler().fit(data["x"])
        # Use exactly the float32 constants that the exported predictor will use.
        mean, scale = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
        x = torch.from_numpy((data["x"] - mean) / scale)
        y = torch.from_numpy(data["y"]).reshape(-1, 1)
    with np.load(artifacts / "validation.npz") as data:
        vx = torch.from_numpy((data["x"] - mean) / scale)
        vy = torch.from_numpy(data["y"]).reshape(-1, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = ConversionMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        model.eval()
        with torch.inference_mode():
            logits = model(vx)
            validation_loss = loss_fn(logits, vy).item()
            auc = float(roc_auc_score(vy.numpy(), torch.sigmoid(logits).numpy()))
        record = {"epoch": epoch, "train_loss": total_loss / len(x),
                  "validation_loss": validation_loss, "validation_roc_auc": auc}
        history.append(record)
        print(json.dumps(record))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    metadata.update({"scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
                     "seed": seed, "epochs": epochs, "best_epoch": best_epoch,
                     "batch_size": batch_size, "learning_rate": 0.001,
                     "torch_version": str(torch.__version__), "architecture": [x.shape[1], 64, 32, 1],
                     "input_contract": "raw float32 features in feature_names order",
                     "output_contract": "float32 purchase probability, shape [batch, 1]"})
    torch.save({"model_state": best_state, "metadata": metadata}, artifacts / "model.pt")
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (artifacts / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Saved validation-selected epoch {best_epoch} to {artifacts / 'model.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.artifacts, args.epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
