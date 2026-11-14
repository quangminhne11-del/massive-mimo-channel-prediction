"""Output-shape smoke checks for the currently registered neural models.

Run with:
    PYTHONPATH=. python tests/test_models_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.utils import build_model, count_trainable_params  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _shape_check(name: str, model, x_shape, y_shape_expected):
    x = torch.randn(*x_shape)
    y = model(x)
    assert tuple(y.shape) == y_shape_expected, (
        f"{name}: expected {y_shape_expected}, got {tuple(y.shape)}"
    )
    return y


def test_all_models():
    model_names = ["pi_dast", "transformer", "cnn", "rnn", "lstm", "gru"]
    for name in model_names:
        cfg = load_config(f"configs/model/{name}.yaml")
        model = build_model(name, cfg)
        n = count_trainable_params(model)
        p_frames = int(cfg["p_frames"])
        q_frames = int(cfg["q_frames"])
        if name in ("pi_dast", "cnn"):
            n_rf, n_m = int(cfg["n_rf"]), int(cfg["n_m"])
            x_shape = (2, p_frames, n_rf, n_m, 2)
            y_expected = (2, q_frames, n_rf, n_m, 2)
        else:
            input_dim = int(cfg["input_dim"])
            x_shape = (2, p_frames, input_dim)
            y_expected = (2, q_frames, input_dim)

        _shape_check(name, model, x_shape, y_expected)
        print(f"  [OK] {name:12s} params={n:>8,}")

    print("\nAll registered models produced the configured output shape.")


if __name__ == "__main__":
    test_all_models()
