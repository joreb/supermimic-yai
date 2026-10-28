"""Numpy-only MLP forward (workers never import torch). weights = [(W, b), ...], W shaped (in, out)."""
import numpy as np


def mlp_forward(weights, x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    for i, (W, b) in enumerate(weights):
        x = x @ W + b
        if i < len(weights) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
    return np.tanh(x)
