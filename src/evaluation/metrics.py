"""Evaluation metrics: per-frame NMSE and zero-forcing sum-rate."""
from __future__ import annotations

from typing import Dict

import torch


def compute_nmse_per_frame(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Args:
        pred, true: (B, Q, ...)  reshapable to (B, Q, D)
    Returns:
        (Q,) NMSE per future frame, averaged over the batch.
    """
    B, Q = pred.shape[0], pred.shape[1]
    pred_flat = pred.reshape(B, Q, -1)
    true_flat = true.reshape(B, Q, -1)
    num = (pred_flat - true_flat).pow(2).sum(dim=-1)
    den = true_flat.pow(2).sum(dim=-1).clamp(min=1e-12)
    return (num / den).mean(dim=0)


def nmse_to_db(nmse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(nmse.clamp(min=1e-12))


def compute_sum_rate(
    He_pred: torch.Tensor,
    He_true: torch.Tensor,
    snr_db: float = 20.0,
) -> Dict[str, torch.Tensor]:
    """Spectral efficiency với EIGEN-BEAMFORMING 1 luồng (SVD) dùng CSI DỰ ĐOÁN.

    Trên kênh hiệu dụng He (N_RF×N_M) sau hybrid-BF, phát 1 luồng theo mode trội:
    tiền mã hoá v1 + bộ thu u1 lấy từ SVD của CSI DỰ ĐOÁN Hp, áp lên kênh THẬT Ht ->
    gain vô hướng g = u1^H·Ht·v1. 1 luồng -> KHÔNG giao thoa liên luồng; sai số dự
    đoán chỉ gây MẤT MÁT beamforming (g < σ1 trần) -> rate tăng theo SNR, bám Perfect
    (khác ZF/MMSE đa luồng bị sàn giao thoa). Trục SNR chuẩn hoá theo gain Perfect.

        rate = log2(1 + |g|²/σ²),   σ² = Pref·10^(-SNR/10),  Pref = E[σ1²]

    He_pred, He_true: (B, Q, N_RF, N_M, 2). Trả 'sum_rate_predicted' / 'sum_rate_ideal'.
    """
    assert He_pred.shape == He_true.shape and He_pred.shape[-1] == 2
    Hp = torch.complex(He_pred[..., 0], He_pred[..., 1])
    Ht = torch.complex(He_true[..., 0], He_true[..., 1])
    N_RF, N_M = Hp.shape[-2], Hp.shape[-1]
    Hp = Hp.reshape(-1, N_RF, N_M)
    Ht = Ht.reshape(-1, N_RF, N_M)

    Up, _, Vhp = torch.linalg.svd(Hp, full_matrices=False)          # mode trội từ dự đoán
    u1p = Up[:, :, 0]                                               # (.,N_RF)
    v1p = Vhp[:, 0, :].conj()                                       # (.,N_M)
    g = torch.einsum("ci,cij,cj->c", u1p.conj(), Ht, v1p)          # gain hiệu dụng (trên kênh thật)
    s1t = torch.linalg.svdvals(Ht)[:, 0]                           # σ1 trần (Perfect)
    Pref = (s1t.abs() ** 2).mean() + 1e-12                         # chuẩn hoá SNR theo Perfect
    sigma2 = Pref * 10.0 ** (-snr_db / 10.0)

    rate_pred = torch.log2(1.0 + g.abs() ** 2 / sigma2).mean()
    rate_ideal = torch.log2(1.0 + s1t.abs() ** 2 / sigma2).mean()
    return {"sum_rate_predicted": rate_pred, "sum_rate_ideal": rate_ideal}
