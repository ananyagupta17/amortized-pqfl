"""Small MNIST CNN and flat-vector parameter (de)serialization."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    """conv5x5(16) -> pool -> conv5x5(32) -> pool -> fc128 -> fc10. ~215k parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_flat(model: nn.Module) -> np.ndarray:
    """All state_dict tensors concatenated into one float32 vector (state_dict order)."""
    return np.concatenate([t.detach().cpu().numpy().ravel() for t in model.state_dict().values()]
                          ).astype(np.float32, copy=False)


def set_flat(model: nn.Module, flat: np.ndarray) -> None:
    state = model.state_dict()
    expected = sum(t.numel() for t in state.values())
    if flat.size != expected:
        raise ValueError(f"parameter vector has {flat.size} values, model needs {expected}")
    offset = 0
    new_state = {}
    for name, t in state.items():
        n = t.numel()
        new_state[name] = torch.from_numpy(flat[offset:offset + n].copy()).view_as(t).to(t.dtype)
        offset += n
    model.load_state_dict(new_state)
