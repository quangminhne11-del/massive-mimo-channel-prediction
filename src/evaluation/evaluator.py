"""End-to-end evaluation of one trained model on a test set."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_nmse_per_frame, compute_sum_rate, nmse_to_db


class Evaluator:
    def __init__(
        self,
        model,
        device,
        normalizer,
        flatten_input: bool = False,
        n_rf: int = 8,
        n_m: int = 4,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.normalizer = normalizer
        self.flatten_input = flatten_input
        self.n_rf = n_rf
        self.n_m = n_m

    def _to_spatial(self, t: torch.Tensor) -> torch.Tensor:
        """Reshape (B, Q, N_RF*N_M*2) -> (B, Q, N_RF, N_M, 2). Pass through if already 5D."""
        if t.dim() == 5:
            return t
        B, Q = t.shape[0], t.shape[1]
        return t.view(B, Q, self.n_rf, self.n_m, 2)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, snr_db_list: List[float]) -> Dict:
        all_preds = []
        all_trues = []

        for batch in tqdm(loader, desc="evaluating"):
            x = batch["history"].to(self.device)
            y = batch["future"].to(self.device)

            if self.flatten_input and x.dim() == 5:
                B = x.shape[0]
                x_in = x.view(B, x.shape[1], -1)
            else:
                x_in = x

            pred = self.model(x_in)
            # pred shape matches y shape from the dataset; both are flat (B,Q,64)
            # for baselines and 5D (B,Q,N_RF,N_M,2) for ST-Mamba.
            all_preds.append(pred.cpu())
            all_trues.append(y.cpu())

        preds = torch.cat(all_preds, dim=0)
        trues = torch.cat(all_trues, dim=0)

        # NMSE works on either layout (flattens internally).
        nmse_per_frame = compute_nmse_per_frame(preds, trues)
        nmse_db_per_frame = nmse_to_db(nmse_per_frame).tolist()
        nmse_avg_db = float(nmse_to_db(nmse_per_frame.mean()).item())

        # Sum-rate requires the 5D (..., N_RF, N_M, 2) layout. Reshape here.
        preds_spatial = self._to_spatial(preds) * self.normalizer.scale
        trues_spatial = self._to_spatial(trues) * self.normalizer.scale

        sum_rates = {}
        for snr in snr_db_list:
            sr = compute_sum_rate(preds_spatial, trues_spatial, snr_db=snr)
            sum_rates[str(snr)] = {
                "predicted": float(sr["sum_rate_predicted"].item()),
                "ideal": float(sr["sum_rate_ideal"].item()),
            }

        return {
            "nmse_per_frame_db": nmse_db_per_frame,
            "nmse_avg_db": nmse_avg_db,
            "sum_rates": sum_rates,
        }
