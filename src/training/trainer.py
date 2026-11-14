"""Generic Trainer for channel-prediction models.

Supports:
  - AdamW + warmup/cosine schedule
  - Mixed precision (FP16) via torch.amp (autocast + GradScaler), when enabled
    and supported by the selected device
  - Gradient clipping
  - Early stopping on validation 'total' loss
  - Checkpoint saving (best + last)
  - TensorBoard logging
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.x
    def _autocast(device_type: str, enabled: bool):
        return autocast(device_type=device_type, enabled=enabled)
    def _grad_scaler(enabled: bool, device_type: str):
        return GradScaler(device=device_type, enabled=enabled)
except Exception:  # legacy fallback
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    def _autocast(device_type: str, enabled: bool):
        return autocast(enabled=enabled)
    def _grad_scaler(enabled: bool, device_type: str):
        return GradScaler(enabled=enabled)

from tqdm import tqdm

from src.training.optimizer import build_optimizer_and_scheduler
from src.utils.logger import Logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        config: dict,
        log_dir: Path,
        ckpt_dir: Path,
        flatten_input: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.flatten_input = flatten_input

        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model, config, steps_per_epoch=len(train_loader)
        )

        # AMP defaults to False. The pure-PyTorch Mamba fallback uses exp() in
        # the SSM scan and is numerically unstable in fp16 (overflow -> NaN).
        # If the user explicitly enables AMP but we are on the fallback,
        # override to False with a warning. AMP can be safely enabled only
        # when the official mamba-ssm CUDA kernels are available.
        use_amp_cfg = bool(config.get("use_amp", False))
        try:
            from src.models.mamba_block import MAMBA_BACKEND  # type: ignore
        except Exception:
            MAMBA_BACKEND = "unknown"
        if use_amp_cfg and MAMBA_BACKEND == "fallback":
            print(
                "[trainer] WARN: AMP disabled because Mamba fallback is "
                "unstable in fp16 (exp overflow)",
                flush=True,
            )
            use_amp_cfg = False
        self.amp_enabled = use_amp_cfg and device.type == "cuda"
        self.device_type = device.type
        self.scaler = _grad_scaler(self.amp_enabled, device.type)

        self.logger = Logger(log_dir)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0  # mutated by load_checkpoint() if --resume

    def load_checkpoint(self, ckpt_path: Path) -> None:
        """Restore training state from a checkpoint so we can continue.

        Supports both old-format checkpoints (only model+optimizer+epoch) and
        new-format ones that additionally include scheduler/scaler/patience.
        Missing pieces fall back to fresh defaults with a warning.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume requested but checkpoint missing: {ckpt_path}")
        print(f"[trainer] resuming from {ckpt_path}", flush=True)
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ck["model_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])

        if "scheduler_state_dict" in ck and ck["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(ck["scheduler_state_dict"])
        else:
            # Old checkpoint -- fast-forward scheduler so LR matches the epoch.
            steps_per_epoch = len(self.train_loader)
            for _ in range(int(ck.get("epoch", 0)) * steps_per_epoch):
                self.scheduler.step()
            print("[trainer] WARN: old checkpoint has no scheduler state; fast-forwarded", flush=True)

        if self.amp_enabled and ck.get("scaler_state_dict") is not None:
            try:
                self.scaler.load_state_dict(ck["scaler_state_dict"])
            except Exception as e:
                print(f"[trainer] WARN: failed to restore AMP scaler ({e}); using fresh", flush=True)

        self.best_val_loss = float(ck.get("best_val_loss", float("inf")))
        self.patience_counter = int(ck.get("patience_counter", 0))
        self.start_epoch = int(ck.get("epoch", -1)) + 1

        print(
            f"[trainer] resumed: start_epoch={self.start_epoch} "
            f"best_val_loss={self.best_val_loss:.6f} "
            f"patience_counter={self.patience_counter}",
            flush=True,
        )

    def train(self) -> float:
        n_epochs = int(self.config["n_epochs"])
        patience = int(self.config.get("patience", 15))
        for epoch in range(self.start_epoch, n_epochs):
            # Loss functions that switch behavior per epoch (e.g. MSE warmup
            # -> NMSE) read this via .set_epoch(). Other losses ignore it.
            if hasattr(self.loss_fn, "set_epoch"):
                self.loss_fn.set_epoch(epoch)
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)
            metrics = {**train_metrics, **val_metrics}
            self.logger.log_metrics(metrics, step=epoch)

            # Track best checkpoint by val/nmse_loss specifically -- val/total
            # can change scale across epochs (e.g. MSE-warmup loss switches to
            # NMSE mid-training and total jumps from ~0.005 to ~0.4), which
            # would falsely freeze the "best" at the warmup epoch. NMSE is the
            # evaluation metric and is comparable across all loss types.
            val_loss = val_metrics.get("val/nmse_loss", val_metrics["val/total_loss"])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, "best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"[trainer] early stop at epoch {epoch}", flush=True)
                    break

            self._save_checkpoint(epoch, "last")

        self.logger.close()
        return self.best_val_loss

    def _prepare_batch(self, batch: Dict) -> tuple:
        x = batch["history"].to(self.device)
        y = batch["future"].to(self.device)
        if self.flatten_input and x.dim() == 5:
            B = x.shape[0]
            x_in = x.view(B, x.shape[1], -1)
        else:
            x_in = x
        return x_in, y

    def _restore_pred_shape(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.flatten_input and y.dim() == 5:
            B = y.shape[0]
            return pred.view(B, y.shape[1], *y.shape[2:])
        return pred

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        sums: Dict[str, float] = {"total": 0.0, "nmse": 0.0, "phase": 0.0, "temporal": 0.0}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"epoch {epoch} [train]")
        for batch in pbar:
            x_in, y = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with _autocast(self.device_type, enabled=self.amp_enabled):
                pred = self.model(x_in)
                pred_for_loss = self._restore_pred_shape(pred, y)
                loss_dict = self.loss_fn(pred_for_loss, y)

            if self.amp_enabled:
                self.scaler.scale(loss_dict["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.scheduler.step()

            for k in sums:
                v = loss_dict[k]
                sums[k] += float(v.item() if isinstance(v, torch.Tensor) else v)
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss_dict['total'].item():.4f}"})

        return {f"train/{k}_loss": (sums[k] / max(1, n_batches)) for k in sums}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        sums: Dict[str, float] = {"total": 0.0, "nmse": 0.0, "phase": 0.0, "temporal": 0.0}
        n_batches = 0

        for batch in tqdm(self.val_loader, desc=f"epoch {epoch} [val]"):
            x_in, y = self._prepare_batch(batch)
            with _autocast(self.device_type, enabled=self.amp_enabled):
                pred = self.model(x_in)
                pred_for_loss = self._restore_pred_shape(pred, y)
                loss_dict = self.loss_fn(pred_for_loss, y)
            for k in sums:
                v = loss_dict[k]
                sums[k] += float(v.item() if isinstance(v, torch.Tensor) else v)
            n_batches += 1

        return {f"val/{k}_loss": (sums[k] / max(1, n_batches)) for k in sums}

    def _save_checkpoint(self, epoch: int, tag: str) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        # Save EVERYTHING needed to resume bit-for-bit: model + optimizer +
        # scheduler + AMP scaler + early-stopping counter. load_checkpoint()
        # restores from these fields.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": (
                    self.scaler.state_dict() if self.amp_enabled else None
                ),
                "best_val_loss": self.best_val_loss,
                "patience_counter": self.patience_counter,
                "config": self.config,
            },
            path,
        )
