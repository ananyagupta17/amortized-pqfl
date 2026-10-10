"""MNIST loading and client data splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


@dataclass
class TensorData:
    x: torch.Tensor  # (n, 1, 28, 28) float32, normalized
    y: torch.Tensor  # (n,) int64

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "TensorData":
        t = torch.as_tensor(idx, dtype=torch.long)
        return TensorData(self.x[t], self.y[t])


def load_mnist(data_dir: str = "data") -> tuple[TensorData, TensorData]:
    """Whole MNIST train/test sets as in-memory tensors (downloaded on first use)."""
    from torchvision.datasets import MNIST

    out = []
    for train in (True, False):
        ds = MNIST(data_dir, train=train, download=True)
        x = ds.data.unsqueeze(1).float().div_(255.0).sub_(MNIST_MEAN).div_(MNIST_STD)
        out.append(TensorData(x, ds.targets.long()))
    return out[0], out[1]


def split_iid(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle and deal indices into k near-equal shards."""
    return [np.sort(s) for s in np.array_split(rng.permutation(n), k)]


def split_dirichlet(labels: np.ndarray, k: int, alpha: float,
                    rng: np.random.Generator) -> list[np.ndarray]:
    """Label-skewed non-IID split: each class is divided across clients ~ Dir(alpha)."""
    shards: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        props = rng.dirichlet(np.full(k, alpha))
        cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
        for shard, part in zip(shards, np.split(idx, cuts)):
            shard.extend(part.tolist())
    return [np.sort(np.array(s, dtype=np.int64)) for s in shards]
