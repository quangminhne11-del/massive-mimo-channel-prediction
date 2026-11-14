"""Read all best checkpoints and TensorBoard logs to summarize training results.

For each experiment dir under results/checkpoints/, prints:
    - exp name + inferred model
    - epoch at which 'best' was saved
    - best val total loss
    - val/nmse_loss at that same epoch (read from the matching TB log)
    - number of trainable params
Then ranks all by val/nmse_loss.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


KNOWN_MODELS = ("st_mamba", "rnn", "lstm", "gru", "transformer")


def infer_model_name(exp_name: str) -> Optional[str]:
    name = exp_name.lower()
    # Try longest first so 'st_mamba' wins over 'mamba' substring
    for cand in sorted(KNOWN_MODELS, key=lambda s: -len(s)):
        if name.startswith(cand + "_") or name == cand:
            return cand
    return None


def count_params_from_state_dict(state_dict: dict) -> int:
    return sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))


def read_tb_val_metrics(log_dir: Path) -> Dict[int, Dict[str, float]]:
    """Returns: epoch -> {tag -> value} for any tag starting with 'val/'."""
    out: Dict[int, Dict[str, float]] = {}
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return out
    # If multiple event files exist (re-runs), merge them; later events overwrite.
    for ef in sorted(event_files, key=lambda p: p.stat().st_mtime):
        try:
            acc = EventAccumulator(str(ef.parent))
            acc.Reload()
            for tag in acc.Tags().get("scalars", []):
                if not tag.startswith("val/"):
                    continue
                for ev in acc.Scalars(tag):
                    out.setdefault(int(ev.step), {})[tag] = float(ev.value)
        except Exception as e:
            print(f"[warn] failed to read {ef}: {e}")
    return out


def summarize_one(exp_dir: Path) -> Optional[dict]:
    ckpt_path = exp_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    exp_name = exp_dir.name
    model = infer_model_name(exp_name) or "unknown"

    epoch = int(ckpt.get("epoch", -1))
    best_val_total = float(ckpt.get("best_val_loss", float("nan")))
    n_params = count_params_from_state_dict(ckpt["model_state_dict"])

    # Pull val/nmse_loss at that epoch from TB logs
    log_dir = PROJECT_ROOT / "results" / "logs" / exp_name
    tb = read_tb_val_metrics(log_dir)
    val_nmse = tb.get(epoch, {}).get("val/nmse_loss", float("nan"))
    val_total_at_epoch = tb.get(epoch, {}).get("val/total_loss", float("nan"))

    return {
        "exp_name": exp_name,
        "model": model,
        "epoch": epoch,
        "best_val_total": best_val_total,
        "val_nmse_loss": val_nmse,
        "val_total_at_epoch": val_total_at_epoch,
        "n_params": n_params,
        "is_nan": (best_val_total != best_val_total),  # NaN check
    }


def main():
    ckpt_root = PROJECT_ROOT / "results" / "checkpoints"
    exps = sorted([p for p in ckpt_root.iterdir() if p.is_dir()])

    summaries: List[dict] = []
    for exp in exps:
        s = summarize_one(exp)
        if s is not None:
            summaries.append(s)

    print()
    print("=" * 92)
    print(f"{'Experiment':40s} {'Model':10s} {'Epoch':>6s} {'Best val_total':>15s} {'val/nmse':>10s} {'Params':>10s}")
    print("-" * 92)
    for s in summaries:
        total_str = "NaN" if s["is_nan"] else f"{s['best_val_total']:.6f}"
        nmse_str = "NaN" if s["val_nmse_loss"] != s["val_nmse_loss"] else f"{s['val_nmse_loss']:.6f}"
        print(
            f"{s['exp_name']:40s} {s['model']:10s} {s['epoch']:>6d} "
            f"{total_str:>15s} {nmse_str:>10s} {s['n_params']:>10,}"
        )
    print("=" * 92)

    # Rank by val/nmse_loss (skip NaN)
    ranked = [s for s in summaries if s["val_nmse_loss"] == s["val_nmse_loss"]]
    ranked.sort(key=lambda s: s["val_nmse_loss"])

    nans = [s for s in summaries if s["val_nmse_loss"] != s["val_nmse_loss"]]

    print()
    print("=== Ranking by val/nmse_loss (lower = better) ===")
    for i, s in enumerate(ranked, 1):
        nmse_db = 10.0 * torch.log10(torch.tensor(max(s["val_nmse_loss"], 1e-12))).item()
        print(
            f"  {i}. {s['exp_name']:40s} | "
            f"val/nmse={s['val_nmse_loss']:.6f} ({nmse_db:6.2f} dB) | "
            f"epoch {s['epoch']}"
        )
    if nans:
        print()
        print("=== NaN / unreadable ===")
        for s in nans:
            print(f"  - {s['exp_name']} (epoch {s['epoch']}, best_val_total={s['best_val_total']})")


if __name__ == "__main__":
    main()
