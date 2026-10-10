"""Hand-rolled FedAvg: local SGD, update (de)serialization, weighted averaging, evaluation."""

from __future__ import annotations

import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pqfl.fl.data import TensorData
from pqfl.fl.model import get_flat, set_flat

_HEADER = struct.Struct(">QQ")  # (num_samples, num_params)


def local_train(model: nn.Module, global_flat: np.ndarray, data: TensorData, *,
                epochs: int, batch_size: int, lr: float, generator: torch.Generator,
                device: torch.device) -> np.ndarray:
    """Start from the global weights, run local SGD, return the new flat weights."""
    set_flat(model, global_flat)
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    x, y = data.x.to(device), data.y.to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(data), generator=generator).to(device)
        for i in range(0, len(data), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
    return get_flat(model)


def serialize_update(weights: np.ndarray, num_samples: int) -> bytes:
    """Plaintext the client encrypts: sample count (FedAvg weight) + float32 weights."""
    w = np.ascontiguousarray(weights, dtype="<f4")
    return _HEADER.pack(num_samples, w.size) + w.tobytes()


def deserialize_update(payload: bytes, expected_params: int) -> tuple[np.ndarray, int]:
    num_samples, num_params = _HEADER.unpack_from(payload)
    if num_params != expected_params or len(payload) != _HEADER.size + 4 * num_params:
        raise ValueError(f"malformed update: {num_params} params, {len(payload)} bytes")
    weights = np.frombuffer(payload, dtype="<f4", offset=_HEADER.size).astype(np.float32)
    if not np.all(np.isfinite(weights)):
        raise ValueError("update contains non-finite values")
    return weights, num_samples


def fedavg(updates: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """Sample-weighted mean of client weight vectors (McMahan et al., 2017)."""
    total = sum(n for _, n in updates)
    if total == 0:
        raise ValueError("no samples to aggregate")
    out = np.zeros_like(updates[0][0], dtype=np.float64)
    for w, n in updates:
        out += (n / total) * w
    return out.astype(np.float32)


@torch.no_grad()
def evaluate(model: nn.Module, data: TensorData, device: torch.device,
             batch_size: int = 2048) -> float:
    model.to(device).eval()
    correct = 0
    for i in range(0, len(data), batch_size):
        x = data.x[i:i + batch_size].to(device)
        y = data.y[i:i + batch_size].to(device)
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(data)
