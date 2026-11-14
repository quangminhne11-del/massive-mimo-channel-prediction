"""Train one model.

Usage:
    python scripts/02_train.py --model st_mamba --train_config full
    python scripts/02_train.py --model lstm     --train_config full
    python scripts/02_train.py --model st_mamba --train_config curriculum
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ChannelDataset  # noqa: E402
from src.data.normalize import ChannelNormalizer  # noqa: E402
from src.losses.combined import build_loss  # noqa: E402
from src.models.utils import build_model, count_trainable_params, is_spatial_model  # noqa: E402
from src.training.curriculum import CurriculumScheduler  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config, merge_configs  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


# Direction A (fair comparison): all 5 models matched to ST-Mamba's only
# stable size (~371K). ST-Mamba's pure-PyTorch fallback collapses above ~80
# d_spatial, so instead of inflating it we bring the baselines DOWN to the
# same budget. ±10% of 371K -> [334K, 408K]; widen slightly for safety.
PARAM_BUDGET_MIN = 330_000
PARAM_BUDGET_MAX = 410_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["pi_dast", "st_mamba", "rnn", "lstm", "gru", "cnn",
                 "transformer", "informer", "hybrid_attention"],
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="full",
        help="name of YAML in configs/train/ (without .yaml)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Override path tới YAML model (cho ablation, vd configs/model/pi_dast_nopad.yaml). "
             "Mặc định configs/model/<model>.yaml.",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from results/checkpoints/<exp_name>/checkpoint_last.pt "
            "(or path passed via --resume_from). Restores model + optimizer + "
            "scheduler + AMP scaler + best/patience state."
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Override resume path (defaults to checkpoint_last.pt under exp_name).",
    )
    args = parser.parse_args()

    set_seed(int(args.seed))

    model_cfg = load_config(args.model_config or f"configs/model/{args.model}.yaml")
    train_cfg = load_config(f"configs/train/{args.train_config}.yaml")
    cfg = merge_configs(model_cfg, train_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}", flush=True)

    data_dir = Path(args.data_dir)
    normalizer = ChannelNormalizer.load(data_dir / "normalize_stats.npz")

    flatten = not is_spatial_model(args.model)

    train_ds = ChannelDataset(data_dir / "train", normalizer=normalizer, flatten_spatial=flatten)
    val_ds = ChannelDataset(data_dir / "val", normalizer=normalizer, flatten_spatial=flatten)

    # Optional curriculum
    # `persistent_workers=True` is critical for long runs: it keeps the DataLoader
    # worker processes alive across epochs instead of respawning + reimporting
    # torch/tensorboard each time. Without this, repeated imports leak memory
    # (~30 MB / epoch with 2 workers) and OOM around epoch 100-130 on a 16 GB
    # machine -- which is exactly what crashed the first 300-epoch Transformer
    # Exp1 run at epoch 129. Only enable when num_workers > 0.
    nw = int(cfg.get("num_workers", 2))
    common_loader_kwargs = dict(
        batch_size=int(cfg["batch_size"]),
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
    )

    if args.train_config == "curriculum":
        stages = [(int(end), tuple(rng)) for end, rng in train_cfg["curriculum_stages"]]
        # Use the full stage that contains all velocities for the loader; the actual
        # curriculum subsetting is left as a per-epoch hook (omitted in this minimal
        # path -- training runs on full train set with the warmup/cosine schedule).
        scheduler = CurriculumScheduler(train_ds, stages)
        train_subset = scheduler.get_subset(int(train_cfg["n_epochs"]))
        train_loader = DataLoader(train_subset, shuffle=True, **common_loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)

    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)

    model = build_model(args.model, model_cfg)
    n_params = count_trainable_params(model)
    print(f"[train] model={args.model} params={n_params:,}", flush=True)
    if not (PARAM_BUDGET_MIN <= n_params <= PARAM_BUDGET_MAX):
        print(
            f"[WARN] params {n_params:,} outside fair budget "
            f"[{PARAM_BUDGET_MIN:,}, {PARAM_BUDGET_MAX:,}] -- "
            f"adjust configs/model/{args.model}.yaml",
            flush=True,
        )

    loss_fn = build_loss(cfg)
    print(f"[train] loss = {cfg.get('loss_type', 'nmse_only')}", flush=True)

    exp_name = args.exp_name or f"{args.model}_{args.train_config}_seed{args.seed}"
    log_dir = Path("results/logs") / exp_name
    ckpt_dir = Path("results/checkpoints") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        config=cfg,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        flatten_input=flatten,
    )

    if args.resume:
        resume_path = Path(args.resume_from) if args.resume_from else (ckpt_dir / "checkpoint_last.pt")
        trainer.load_checkpoint(resume_path)

    best_val = trainer.train()
    print(f"[train] done. best val total_loss = {best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
