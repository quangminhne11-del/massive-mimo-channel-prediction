"""Evaluate saved channel-prediction models and write metrics and figures.

The script loads data from --data_dir and checkpoints selected by --exp_suffix.
Optional sections include SNR/velocity sweeps, BER, complexity, and ablations.
It writes generated summaries and plots beneath results/.

Examples:
    python evaluate.py --data_dir data_sionna --exp_suffix exp1_seed42
    python evaluate.py --data_dir data_sionna --exp_suffix exp1_seed42 --skip_sweeps --skip_ber
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _s in ("stdout", "stderr"):  # console Windows cp1252 -> ép UTF-8 cho tiếng Việt
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.channel_generator import ChannelConfig  # noqa: E402
from src.data.normalize import ChannelNormalizer  # noqa: E402
from src.evaluation.metrics import compute_nmse_per_frame, compute_sum_rate, nmse_to_db  # noqa: E402
from src.models.pad import pad_predict  # noqa: E402
from src.models.kalman import kalman_predict  # noqa: E402
from src.models.utils import build_model, count_trainable_params, is_spatial_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Cấu hình hiển thị
# ---------------------------------------------------------------------------
# Lineup gọn: ĐỀ XUẤT (DAST) + baseline Transformer/LSTM/GRU/CNN. RNN bỏ khỏi
# biểu đồ cho trong. PAD/Kalman/No-Pred/Perfect = tham chiếu (không train).
MODELS = ["pi_dast", "transformer", "lstm", "gru", "cnn"]
DISP = {
    "pi_dast": "DAST",                      # đề xuất (= pi_dast_nophys, bản xịn nhất)
    "transformer": "Transformer", "lstm": "LSTM", "gru": "GRU", "cnn": "CNN",
    "pad": "PAD (classical)", "kalman": "Kalman (classical)",
    "no_pred": "No-Prediction (Outdated)", "perfect": "Perfect CSI",
}
STYLE = {  # color, linestyle, marker, linewidth
    "pi_dast":          ("#d62728", "-",  "o", 2.6),   # ĐỀ XUẤT -- đỏ đậm
    "transformer":      ("#8c564b", "-",  "P", 1.6),
    "lstm":             ("#2ca02c", "-",  "v", 1.4),
    "gru":              ("#ff7f0e", "-",  "^", 1.4),
    "cnn":              ("#bcbd22", "-",  "h", 1.4),
    "pad":              ("#7f7f7f", "--", "x", 1.6),
    "kalman":           ("#17becf", "--", "D", 1.6),
    "no_pred":          ("#555555", ":",  "+", 1.6),
    "perfect":          ("#000000", ":",  None, 2.2),
}
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TBL_DIR = PROJECT_ROOT / "results" / "tables"

SNR_LIST = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]      # NMSE/Sum-rate (mở xuống 0 dB để lộ vùng động)
SNR_LIST_BER = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # BER waterfall (QPSK 1 luồng)
VEL_LIST = [30.0, 50.0, 70.0, 90.0, 120.0]
P_LIST = [30, 60, 100]                          # complexity scaling


def _plot(ax, x, y, method, **kw):
    c, ls, mk, lw = STYLE[method]
    ax.plot(x, y, color=c, linestyle=ls, marker=mk, linewidth=lw, label=DISP[method], **kw)


# ---------------------------------------------------------------------------
# Tải model
# ---------------------------------------------------------------------------
def load_models(exp_suffix: str, device):
    loaded = {}
    for m in MODELS:
        ck = PROJECT_ROOT / "results" / "checkpoints" / f"{m}_{exp_suffix}" / "checkpoint_best.pt"
        if not ck.exists():
            alt = PROJECT_ROOT / "results" / "checkpoints" / f"{m}_full_seed42" / "checkpoint_best.pt"
            if alt.exists():
                ck = alt
            else:
                print(f"[load] BỎ QUA {m}: thiếu checkpoint")
                continue
        cfg = load_config(str(PROJECT_ROOT / "configs" / "model" / f"{m}.yaml"))
        model = build_model(m, cfg)
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model_state_dict"])
        model.to(device).eval()
        loaded[m] = (model, not is_spatial_model(m), count_trainable_params(model))
        print(f"[load] {DISP.get(m, m):<14} params={loaded[m][2]:,}  <- {ck.parent.name}")
    return loaded


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_saved_samples(test_dir: Path, limit=None):
    files = sorted(test_dir.glob("sample_*.npz"))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        d = np.load(f)
        s = {"He_history": d["He_history"].astype(np.float32),
             "He_future": d["He_future"].astype(np.float32)}
        if "He_history_clean" in d.files:
            s["He_history_clean"] = d["He_history_clean"].astype(np.float32)
        out.append(s)
    return out


def renoise_samples(samples, snr_db, seed=0):
    """Lấy He_history_clean + AWGN ở snr_db -> bộ test cùng kênh, khác SNR quan sát.

    Nhiễu calibrate theo công suất kênh sạch (cùng công thức generator Sionna).
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in samples:
        hc = s["He_history_clean"]
        sig = float(np.mean(np.sum(hc ** 2, axis=-1)))      # E|He|^2 (re^2+im^2)
        std = math.sqrt(sig / (10.0 ** (snr_db / 10.0)) / 2.0)
        noisy = hc + std * rng.standard_normal(hc.shape).astype(np.float32)
        out.append({"He_history": noisy.astype(np.float32), "He_future": s["He_future"]})
    return out


@torch.no_grad()
def collect_predictions(method, models, samples, normalizer, device, batch=64):
    """Trả về (preds_phys, trues_phys) shape (N,Q,nrf,nm,2) ở thang VẬT LÝ."""
    trues = np.stack([s["He_future"] for s in samples]).astype(np.float32)
    Q = trues.shape[1]
    if method == "perfect":
        return trues.copy(), trues
    if method == "no_pred":
        hist = np.stack([s["He_history"] for s in samples]).astype(np.float32)
        preds = np.repeat(hist[:, -1:], Q, axis=1)  # lặp frame cuối
        return preds.astype(np.float32), trues
    if method == "pad":
        hist = np.stack([s["He_history"] for s in samples]).astype(np.float32)
        return pad_predict(hist, q_frames=Q).astype(np.float32), trues
    if method == "kalman":
        hist = np.stack([s["He_history"] for s in samples]).astype(np.float32)
        return kalman_predict(hist, q_frames=Q).astype(np.float32), trues
    model, flatten, _ = models[method]
    preds = []
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        hist = np.stack([normalizer.transform(s["He_history"]) for s in chunk]).astype(np.float32)
        x = torch.from_numpy(hist).to(device)
        if flatten:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        p = model(x).reshape(len(chunk), Q, trues.shape[2], trues.shape[3], 2)
        preds.append(p.cpu().numpy() * normalizer.scale)  # denormalize
    return np.concatenate(preds, 0).astype(np.float32), trues


def nmse_per_frame_db(preds, trues):
    pf = compute_nmse_per_frame(torch.from_numpy(preds), torch.from_numpy(trues))
    return nmse_to_db(pf).tolist(), float(nmse_to_db(pf.mean()).item())


# NMSE hiển thị dạng TUYẾN TÍNH (tỉ số), a = 10^(dB/10). Thấp = tốt; >1 = tệ hơn No-Prediction.
def nmse_lin(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


NMSE_YLAB = "NMSE (dB)"   # dB = 10·log10(NMSE) -- chuẩn ngành, tách rõ model


# ---------------------------------------------------------------------------
# BER QPSK với eigen-beamforming 1 luồng (SVD) từ CSI dự đoán
# ---------------------------------------------------------------------------
def _dominant_modes(H):
    """SVD mode trội: H (Nc,nrf,nm) -> u1 (Nc,nrf), v1 (Nc,nm), s1 (Nc,)."""
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    return U[:, :, 0], np.conj(Vh[:, 0, :]), S[:, 0]


def compute_ber(preds, trues, snr_list, n_sym=256, seed=0):
    """BER QPSK với EIGEN-BEAMFORMING 1 luồng (SVD) từ CSI DỰ ĐOÁN.

    Trên kênh hiệu dụng He (nrf×nm), phát 1 luồng theo mode trội: tiền mã hoá v1 +
    bộ thu u1 lấy từ SVD của Hp DỰ ĐOÁN, áp lên kênh THẬT Ht -> gain vô hướng
    g = u1^H·Ht·v1. 1 luồng -> KHÔNG giao thoa liên luồng; sai số CSI chỉ gây MẤT MÁT
    beamforming (lệch SNR) -> BER WATERFALL (không sàn cứng như ZF/MMSE đa luồng).
    Trục SNR chuẩn hoá theo gain Perfect (Pref=E[σ1²]) nên Perfect-CSI = QPSK AWGN.
    preds/trues: (N,Q,nrf,nm,2) vật lý. n_sym lớn để phân giải BER thấp.
    """
    rng = np.random.default_rng(seed)
    Hp = (preds[..., 0] + 1j * preds[..., 1]).reshape(-1, preds.shape[-3], preds.shape[-2])  # (Nc,nrf,nm)
    Ht = (trues[..., 0] + 1j * trues[..., 1]).reshape(-1, trues.shape[-3], trues.shape[-2])
    u1p, v1p, _ = _dominant_modes(Hp)                          # beamformer từ DỰ ĐOÁN
    _, _, s1t = _dominant_modes(Ht)                            # σ1 trần (Perfect)
    g = np.einsum("ci,cij,cj->c", np.conj(u1p), Ht, v1p)       # gain hiệu dụng trên kênh THẬT
    Pref = float(np.mean(s1t ** 2)) + 1e-12                    # chuẩn hoá SNR theo Perfect
    Nc = g.shape[0]
    b = rng.integers(0, 2, size=(Nc, n_sym, 2))               # 2 bit/symbol (QPSK)
    s = ((1 - 2 * b[..., 0]) + 1j * (1 - 2 * b[..., 1])) / math.sqrt(2.0)  # QPSK unit-energy
    gz = g[:, None]
    out = []
    for snr in snr_list:
        sigma2 = Pref * 10.0 ** (-snr / 10.0)
        noise = math.sqrt(sigma2 / 2.0) * (
            rng.standard_normal((Nc, n_sym)) + 1j * rng.standard_normal((Nc, n_sym)))
        shat = (gz * s + noise) / gz                          # coherent (chia gain phức)
        bh = np.stack([(shat.real < 0).astype(np.int64), (shat.imag < 0).astype(np.int64)], axis=-1)
        out.append(float(np.mean(bh != b)))
    return out


# ---------------------------------------------------------------------------
# PHẦN 2.1: Learning curves từ TensorBoard
# ---------------------------------------------------------------------------
def fig_learning_curves(exp_suffix):
    logdir = PROJECT_ROOT / "results" / "logs" / f"pi_dast_{exp_suffix}"
    if not logdir.exists():
        print(f"[fig] BỎ QUA learning_curves: thiếu {logdir}")
        return
    try:
        from tbparse import SummaryReader
        df = SummaryReader(str(logdir)).scalars
        def curve(tag):
            d = df[df["tag"] == tag].sort_values("step")
            return d["step"].to_numpy(), d["value"].to_numpy()
    except Exception as e:
        print(f"[fig] BỎ QUA learning_curves (tbparse lỗi: {e})")
        return
    plt.figure(figsize=(8, 5))
    got = False
    for tag, lab, col in (("train/nmse_loss", "Train NMSE", "#1f77b4"),
                          ("val/nmse_loss", "Val NMSE", "#d62728")):
        xs, ys = curve(tag)
        if len(xs):
            plt.plot(xs, ys, label=lab, color=col, linewidth=1.8); got = True
    if not got:
        plt.close(); print("[fig] BỎ QUA learning_curves: không thấy tag nmse_loss"); return
    plt.xlabel("Epoch"); plt.ylabel("NMSE loss")
    plt.yscale("log")
    plt.title("Learning Curves -- DAST (Train vs Val)")
    plt.legend(); plt.grid(True, alpha=0.3, which="both")
    p = FIG_DIR / "learning_curves_pi_dast.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig] {p.name}")


# ---------------------------------------------------------------------------
# PHẦN 2.2 + 3: các biểu đồ trên test set đã lưu (v30, v50)
# ---------------------------------------------------------------------------
def eval_scenarios(models, normalizer, device, data_dir, do_ber, summary):
    """Tính preds 1 lần/scenario rồi vẽ: nmse_vs_frame, sumrate_vs_snr, ber_vs_snr."""
    for sc, vel in (("v30", 30), ("v50", 50), ("v90", 90), ("v120", 120)):
        test_dir = data_dir / f"test_{sc}"
        if not test_dir.exists():
            print(f"[scen] BỎ QUA {sc}: thiếu {test_dir}"); continue
        samples = load_saved_samples(test_dir)
        methods = MODELS + ["pad", "kalman", "no_pred"]
        store = {}
        for m in methods:
            if m in MODELS and m not in models:
                continue
            preds, trues = collect_predictions(m, models, samples, normalizer, device)
            pf_db, avg_db = nmse_per_frame_db(preds, trues)
            store[m] = {"preds": preds, "trues": trues, "nmse_per_frame_db": pf_db, "nmse_avg_db": avg_db}
            print(f"[scen] {sc} {DISP.get(m, m):<22} NMSE_avg={avg_db:.2f} dB")
        summary.setdefault("scenarios", {})[sc] = {
            m: {"nmse_per_frame_db": store[m]["nmse_per_frame_db"],
                "nmse_avg_db": store[m]["nmse_avg_db"]} for m in store}

        # --- NMSE vs frame ---
        plt.figure(figsize=(8, 5)); ax = plt.gca()
        for m in methods:
            if m in store:
                _plot(ax, range(1, len(store[m]["nmse_per_frame_db"]) + 1),
                      store[m]["nmse_per_frame_db"], m)
        ax.axhline(0.0, color="k", ls=":", lw=0.8)   # 0 dB = NMSE 1.0 (mức 'không dự đoán')
        ax.set_xlabel("Khung dự đoán (x 0.125 ms)"); ax.set_ylabel(NMSE_YLAB)
        ax.set_title(f"NMSE theo khung dự đoán -- v={vel} km/h, SNR=20 dB")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        p = FIG_DIR / f"nmse_vs_frame_{sc}.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[fig] {p.name}")

        # --- Sum-rate vs SNR (MMSE receiver) ---
        sr_store = {}
        ideal_curve = None
        for m in methods:
            if m not in store:
                continue
            pe = torch.from_numpy(store[m]["preds"]); tr = torch.from_numpy(store[m]["trues"])
            ys = []
            ideal = []
            for snr in SNR_LIST:
                r = compute_sum_rate(pe, tr, snr_db=snr)
                ys.append(float(r["sum_rate_predicted"])); ideal.append(float(r["sum_rate_ideal"]))
            sr_store[m] = ys
            ideal_curve = ideal
        plt.figure(figsize=(8, 5)); ax = plt.gca()
        if ideal_curve:
            _plot(ax, SNR_LIST, ideal_curve, "perfect")
        for m in methods:
            if m in sr_store:
                _plot(ax, SNR_LIST, sr_store[m], m)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Spectral Efficiency / Sum-rate (bps/Hz)")
        ax.set_title(f"Spectral Efficiency theo SNR (eigen-beamforming 1 luồng) -- v={vel} km/h")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        p = FIG_DIR / f"sumrate_vs_snr_{sc}.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[fig] {p.name}")
        summary["scenarios"][sc]["sumrate_vs_snr"] = {"snr": SNR_LIST, "ideal": ideal_curve,
                                                      "curves": sr_store}

        # --- BER vs SNR (QPSK, eigen-beamforming 1 luồng) ---
        if do_ber:
            ber_store = {}
            for m in methods + ["perfect"]:
                if m in MODELS and m not in models:
                    continue
                preds = store[m]["preds"] if m in store else None
                if m == "perfect":
                    preds, trues = collect_predictions("perfect", models, samples, normalizer, device)
                    trues_b = trues
                else:
                    trues_b = store[m]["trues"]
                ber_store[m] = compute_ber(preds, trues_b, SNR_LIST_BER, seed=0)
                print(f"[ber]  {sc} {DISP.get(m, m):<22} " + " ".join(f"{b:.4f}" for b in ber_store[m]))
            plt.figure(figsize=(8, 5)); ax = plt.gca()
            for m in ["perfect"] + methods:
                if m in ber_store:
                    y = np.clip(ber_store[m], 1e-6, 1.0)
                    _plot(ax, SNR_LIST_BER, y, m)
            ax.set_yscale("log"); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER (QPSK)")
            ax.set_title(f"Bit Error Rate theo SNR (QPSK, beamforming) -- v={vel} km/h")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
            p = FIG_DIR / f"ber_vs_snr_{sc}.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            print(f"[fig] {p.name}")
            summary["scenarios"][sc]["ber_vs_snr"] = {"snr": SNR_LIST_BER, "curves": ber_store}

    # --- bar NMSE trung bình ---
    scs = [s for s in ("v30", "v50", "v90", "v120") if s in summary.get("scenarios", {})]
    if scs:
        present = [m for m in MODELS + ["pad", "kalman", "no_pred"]
                   if any(m in summary["scenarios"][s] for s in scs)]
        x = np.arange(len(present)); w = 0.8 / len(scs)
        plt.figure(figsize=(10, 5))
        for j, sc in enumerate(scs):
            vals = [summary["scenarios"][sc].get(m, {}).get("nmse_avg_db", np.nan) for m in present]
            plt.bar(x + (j - (len(scs) - 1) / 2) * w, vals, w, label=f"v={sc[1:]} km/h")
        plt.axhline(0.0, color="k", ls=":", lw=0.8)
        plt.xticks(x, [DISP[m] for m in present], rotation=15, fontsize=8)
        plt.ylabel(NMSE_YLAB); plt.title("So sánh NMSE trung bình (dB)")
        plt.legend(); plt.grid(True, axis="y", alpha=0.3)
        p = FIG_DIR / "nmse_avg_bar.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[fig] {p.name}")


# ---------------------------------------------------------------------------
# PHẦN 3: NMSE vs SNR & vs Velocity (từ test set Sionna đã lưu)
# ---------------------------------------------------------------------------
def fig_nmse_vs_snr(models, normalizer, device, n_sweep, summary, data_dir):
    for sc, vel in (("v30", 30.0), ("v50", 50.0), ("v90", 90.0), ("v120", 120.0)):
        base = load_saved_samples(data_dir / f"test_{sc}", limit=n_sweep)
        if not base or "He_history_clean" not in base[0]:
            print(f"[snr] BỎ QUA {sc}: thiếu test set hoặc He_history_clean"); continue
        snr_samples = {snr: renoise_samples(base, snr, seed=200000 + int(snr)) for snr in SNR_LIST}
        curves = {}
        for m in MODELS + ["pad", "kalman", "no_pred"]:
            if m in MODELS and m not in models:
                continue
            ys = []
            for snr in SNR_LIST:
                pr, tr = collect_predictions(m, models, snr_samples[snr], normalizer, device)
                ys.append(nmse_per_frame_db(pr, tr)[1])
            curves[m] = ys
            print(f"[snr] {sc} {DISP.get(m, m):<22} " + " ".join(f"{v:.1f}" for v in ys))
        plt.figure(figsize=(8, 5)); ax = plt.gca()
        for m in MODELS + ["pad", "kalman", "no_pred"]:
            if m in curves:
                _plot(ax, SNR_LIST, curves[m], m)
        ax.axhline(0.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel(NMSE_YLAB)
        ax.set_title(f"NMSE theo SNR -- v={vel:.0f} km/h")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        p = FIG_DIR / f"nmse_vs_snr_{sc}.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[fig] {p.name}")
        summary.setdefault("nmse_vs_snr", {})[sc] = {"snr": SNR_LIST, "curves": curves}


def fig_nmse_vs_velocity(models, normalizer, device, n_sweep, summary, data_dir):
    vel_samples = {v: load_saved_samples(data_dir / f"test_v{int(v)}", limit=n_sweep)
                   for v in VEL_LIST}
    vel_samples = {v: s for v, s in vel_samples.items() if s}
    if not vel_samples:
        print("[vel] BỎ QUA: không có test_v* trong data_dir"); return
    vels = sorted(vel_samples)
    curves = {}
    for m in MODELS + ["pad", "kalman", "no_pred"]:
        if m in MODELS and m not in models:
            continue
        ys = []
        for v in vels:
            pr, tr = collect_predictions(m, models, vel_samples[v], normalizer, device)
            ys.append(nmse_per_frame_db(pr, tr)[1])
        curves[m] = ys
        print(f"[vel] {DISP.get(m, m):<22} " + " ".join(f"{v:.1f}" for v in ys))
    plt.figure(figsize=(8.5, 5)); ax = plt.gca()
    for m in MODELS + ["pad", "kalman", "no_pred"]:
        if m in curves:
            _plot(ax, vels, curves[m], m)
    ax.axhline(0.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Vận tốc người dùng (km/h)"); ax.set_ylabel(NMSE_YLAB)
    ax.set_title("NMSE theo vận tốc -- SNR=20 dB")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    p = FIG_DIR / "nmse_vs_velocity.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig] {p.name}")
    summary["nmse_vs_velocity"] = {"velocity": vels, "curves": curves}


# ---------------------------------------------------------------------------
# PHẦN 4: Complexity & Efficiency (Params / FLOPs / Latency @ P=30,60,100)
# ---------------------------------------------------------------------------
def _build_for_P(m, P, device):
    cfg = load_config(str(PROJECT_ROOT / "configs" / "model" / f"{m}.yaml"))
    cfg = {k: v for k, v in cfg.items() if k != "model_name"}
    cfg["p_frames"] = P
    model = build_model(m, cfg).to(device).eval()
    flatten = not is_spatial_model(m)
    cc = ChannelConfig()
    if flatten:
        dummy = torch.randn(1, P, cc.n_rf * cc.n_m * 2, device=device)
    else:
        dummy = torch.randn(1, P, cc.n_rf, cc.n_m, 2, device=device)
    return model, dummy


def _latency_ms(model, dummy, device, iters=50):
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _flops_M(model, dummy):
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e6   # MACs -> FLOPs (M)
    except Exception:
        return None


def complexity_table(models, device, summary):
    rows = {}          # m -> {P -> (params, flops_M, lat_ms)}
    for m in MODELS:
        if m not in models:
            continue
        rows[m] = {}
        for P in P_LIST:
            try:
                model, dummy = _build_for_P(m, P, device)
                params = count_trainable_params(model)
                flops = _flops_M(model, dummy)
                lat = _latency_ms(model, dummy, device)
                rows[m][P] = (params, flops, lat)
                fl = f"{flops:.1f}" if flops is not None else "n/a"
                print(f"[cplx] {DISP.get(m, m):<14} P={P:<3d} params={params/1e3:.1f}K FLOPs={fl}M lat={lat:.2f}ms")
            except Exception as e:
                print(f"[cplx] {m} P={P} LỖI: {e}")
    summary["complexity"] = {m: {str(P): {"params": rows[m][P][0], "flops_M": rows[m][P][1],
                                          "latency_ms": rows[m][P][2]} for P in rows[m]}
                             for m in rows}

    # --- bảng markdown ---
    lines = ["# Complexity & Efficiency Analysis\n",
             "FLOPs ước lượng bằng thop (lưu ý: vòng scan thuần-Python của Mamba fallback "
             "KHÔNG được thop đếm -> FLOPs ST-Mamba bị thấp hơn thực tế; latency mới phản ánh đúng).\n",
             "Latency = ms / forward (batch=1) trên " + ("GPU" if device.type == "cuda" else "CPU") + ".\n"]
    for P in P_LIST:
        lines.append(f"\n## P = {P} khung lịch sử\n")
        lines.append("| Model | Params (K) | FLOPs (M) | Latency (ms) |")
        lines.append("|---|---|---|---|")
        for m in MODELS:
            if m in rows and P in rows[m]:
                pa, fl, la = rows[m][P]
                fls = f"{fl:.1f}" if fl is not None else "n/a"
                lines.append(f"| {DISP[m]} | {pa/1e3:.1f} | {fls} | {la:.2f} |")
    (TBL_DIR / "complexity_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[tbl] complexity_table.md")

    # --- bar params + latency @ P=10 (cấu hình CHÍNH của đề tài) ---
    P_BAR = 10
    present = [m for m in MODELS if m in rows]
    names = [DISP[m] for m in present]
    pk, lat = [], []
    for m in present:
        mdl, dmy = _build_for_P(m, P_BAR, device)
        pk.append(count_trainable_params(mdl) / 1e3)
        lat.append(_latency_ms(mdl, dmy, device))
    fig, ax1 = plt.subplots(figsize=(9, 5)); x = np.arange(len(present)); w = 0.4
    ax1.bar(x - w / 2, pk, w, color="#4c72b0", label="#Params (K)")
    ax1.set_ylabel("Số tham số (nghìn)", color="#4c72b0")
    ax2 = ax1.twinx(); ax2.bar(x + w / 2, lat, w, color="#dd8452", label="Latency (ms)")
    ax2.set_ylabel("Latency (ms)", color="#dd8452")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15, fontsize=8)
    plt.title(f"Hiệu năng: số tham số & độ trễ inference (P={P_BAR})")
    p = FIG_DIR / "efficiency_bar.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig] {p.name}")

    # --- latency vs P (O(N) vs O(N^2)) ---
    plt.figure(figsize=(8, 5)); ax = plt.gca()
    for m in present:
        ys = [rows[m][P][2] for P in P_LIST]
        c, ls, mk, lw = STYLE[m]
        ax.plot(P_LIST, ys, color=c, marker=mk, linewidth=lw, label=DISP[m])
    ax.set_xlabel("Độ dài chuỗi lịch sử P"); ax.set_ylabel("Latency (ms)")
    ax.set_title("Độ trễ inference theo độ dài chuỗi lịch sử P")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    p = FIG_DIR / "latency_vs_P.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig] {p.name}")


# ---------------------------------------------------------------------------
# PHẦN 4: Bảng Ablation DAST -- đóng góp DUY NHẤT được kiểm chứng = Doppler-aware
# encoding. Cả 2 nhánh train CÙNG recipe NMSE-only (exp1), khác đúng 1 biến
# use_doppler_enc -> ablation single-variable, sạch. (Physics-loss đã bỏ: thử
# nghiệm cho thấy không cải thiện ở mức 1 seed -> không claim.)
# ---------------------------------------------------------------------------
def ablation_table(normalizer, device, data_dir, summary):
    ABL = [
        ("DAST (full)",            "pi_dast_exp1_seed42",      "configs/model/pi_dast.yaml"),
        ("w/o gate (fixed blend)", "pi_dast_fixedgate_seed42", "configs/model/pi_dast_fixedgate.yaml"),
        ("w/o PAD-prior (pure NN)","pi_dast_nopad_seed42",     "configs/model/pi_dast_nopad.yaml"),
        ("w/o Doppler-enc",        "pi_dast_nodoppler_seed42", "configs/model/pi_dast_nodoppler.yaml"),
    ]
    rows = []
    for label, exp, cfgpath in ABL:
        ck = PROJECT_ROOT / "results" / "checkpoints" / exp / "checkpoint_best.pt"
        if not ck.exists():
            print(f"[abl] BỎ QUA {label}: thiếu checkpoint {exp}")
            continue
        model = build_model("pi_dast", load_config(str(PROJECT_ROOT / cfgpath)))
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model_state_dict"])
        model.to(device).eval()
        tmp = {"pi_dast": (model, False, count_trainable_params(model))}
        res = {}
        for sc in ("v30", "v50"):
            td = data_dir / f"test_{sc}"
            if not td.exists():
                continue
            samples = load_saved_samples(td)
            preds, trues = collect_predictions("pi_dast", tmp, samples, normalizer, device)
            res[sc] = nmse_per_frame_db(preds, trues)[1]
        rows.append((label, res))
        print(f"[abl] {label:22s} v30={res.get('v30', float('nan')):.2f}dB "
              f"v50={res.get('v50', float('nan')):.2f}dB")
    if not rows:
        print("[abl] chưa có checkpoint ablation -> bỏ bảng ablation (train pi_dast_* trước)")
        return
    lines = ["# Ablation Study — DAST\n",
             "NMSE dạng dB (thấp = tốt). BER xem hình waterfall riêng.\n",
             "| Variant | NMSE v30 (dB) | NMSE v50 (dB) |", "|---|---|---|"]
    for label, res in rows:
        lines.append(f"| {label} | {res.get('v30', float('nan')):.2f} | "
                     f"{res.get('v50', float('nan')):.2f} |")
    (TBL_DIR / "ablation_table.md").write_text("\n".join(lines), encoding="utf-8")
    summary["ablation"] = [{"variant": l, **r} for l, r in rows]
    print("[tbl] ablation_table.md")


# ---------------------------------------------------------------------------
# Phụ: kênh dự đoán vs thật
# ---------------------------------------------------------------------------
def fig_pred_vs_true(models, normalizer, device, data_dir, scenario="v30"):
    samples = load_saved_samples(data_dir / f"test_{scenario}")
    if not samples or "pi_dast" not in models:
        return
    show = [m for m in ("pi_dast", "transformer") if m in models]
    allpred = {m: collect_predictions(m, models, samples, normalizer, device)[0] for m in show}
    trues = collect_predictions("pi_dast", models, samples, normalizer, device)[1]
    # chọn SAMPLE TỐT NHẤT = nơi DAST có NMSE per-sample thấp nhất
    pr = allpred["pi_dast"]
    num = ((pr - trues) ** 2).reshape(pr.shape[0], -1).sum(1)
    den = (trues ** 2).reshape(trues.shape[0], -1).sum(1) + 1e-12
    nmse_db = 10.0 * np.log10(num / den)
    best = int(np.argmin(nmse_db))
    true_f = trues[best]; Q = true_f.shape[0]
    preds = {m: allpred[m][best] for m in show}
    # vẽ PHẦN THỰC; chọn 4 tap mà DAST bám sát nhất Ở PHẦN THỰC (đúng cái hiển thị)
    err_re = ((preds["pi_dast"][..., 0] - true_f[..., 0]) ** 2).sum(axis=0)   # (nrf,nm)
    flat = np.argsort(err_re, axis=None)[:4]
    pairs = [(int(i // err_re.shape[1]), int(i % err_re.shape[1])) for i in flat]
    fr = np.arange(1, Q + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (rf, mm) in zip(axes.ravel(), pairs):
        ax.plot(fr, true_f[:, rf, mm, 0], "k-o", label="Thật (Re)", linewidth=2)
        for m in show:
            c = STYLE[m][0]
            ax.plot(fr, preds[m][:, rf, mm, 0], "--s", color=c, label=f"{DISP[m]}")
        ax.set_title(f"RF={rf}, UE-ant={mm}"); ax.grid(True, alpha=0.3); ax.set_xlabel("Khung tương lai")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Kênh dự đoán vs kênh thật (phần thực) -- sample tốt nhất "
                 f"(DAST NMSE={nmse_db[best]:.1f} dB), v={scenario[1:]} km/h")
    p = FIG_DIR / f"pred_vs_true_{scenario}.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig] {p.name}  (best sample idx={best}, DAST NMSE={nmse_db[best]:.1f} dB)")


# ---------------------------------------------------------------------------
# Explainability: cổng Doppler g học được (heatmap vận tốc × khung tương lai)
# ---------------------------------------------------------------------------
def fig_doppler_gate(models, normalizer, device, data_dir, n=300):
    if "pi_dast" not in models:
        return
    m = models["pi_dast"][0]
    if not (getattr(m, "use_pad_prior", False) and getattr(m, "use_gate", False)):
        print("[fig] BỎ QUA gate: model không có cổng Doppler"); return
    vels, G = [], []
    for v in VEL_LIST:
        td = data_dir / f"test_v{int(v)}"
        if not td.exists():
            continue
        sm = load_saved_samples(td, limit=n)
        x = torch.from_numpy(np.stack([normalizer.transform(s["He_history"]) for s in sm]))
        x = x.reshape(x.shape[0], x.shape[1], -1).to(device)
        with torch.no_grad():
            h = m._encode_spatial(x)
            prof = torch.fft.rfft(h, dim=1).abs()[:, 1:].mean(dim=-1)
            g = torch.sigmoid(m.gate(prof)).mean(0).cpu().numpy()   # (Q,)
        vels.append(int(v)); G.append(g)
    if not G:
        return
    G = np.array(G)                                                # (n_vel, Q)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(G, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(G.shape[1])); ax.set_xticklabels([f"f{i+1}" for i in range(G.shape[1])])
    ax.set_yticks(range(len(vels))); ax.set_yticklabels([f"{v}" for v in vels])
    ax.set_xlabel("Khung dự đoán tương lai"); ax.set_ylabel("Vận tốc (km/h)")
    ax.set_title("Cổng Doppler g")
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center",
                    color="white" if G[i, j] > 0.55 or G[i, j] < 0.2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="g")
    p = FIG_DIR / "doppler_gate.png"; fig.tight_layout(); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig] {p.name}")


# ---------------------------------------------------------------------------
def _stitch_2x2(paths_in_order, out_path, layout="v30/v50 | v90/v120"):
    """Ghép 4 PNG thành lưới 2x2, mỗi ô GIỮ NGUYÊN kích thước gốc (không scale)
    -> ảnh ghép = 2x rộng, 2x cao so với 1 hình lẻ. Thứ tự: [0,1] hàng trên,
    [2,3] hàng dưới (trái->phải)."""
    from PIL import Image
    miss = [p.name for p in paths_in_order if not p.exists()]
    if miss:
        print(f"[grid] thiếu {miss} -> bỏ ghép"); return
    imgs = [Image.open(p).convert("RGB") for p in paths_in_order]
    cw = max(im.width for im in imgs)            # ô = kích thước hình lớn nhất (để lưới đều)
    ch = max(im.height for im in imgs)
    canvas = Image.new("RGB", (cw * 2, ch * 2), "white")
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, 2)
        canvas.paste(im, (c * cw + (cw - im.width) // 2, r * ch + (ch - im.height) // 2))
    canvas.save(out_path)
    print(f"[fig] {out_path.name} ({canvas.width}x{canvas.height}) -- lưới 2x2 {layout}")


def fig_ber_grid(scenarios=("v30", "v50", "v90", "v120")):
    _stitch_2x2([FIG_DIR / f"ber_vs_snr_{sc}.png" for sc in scenarios],
                FIG_DIR / "ber_vs_snr_grid.png")


def fig_nmse_snr_grid(scenarios=("v30", "v50", "v90", "v120")):
    _stitch_2x2([FIG_DIR / f"nmse_vs_snr_{sc}.png" for sc in scenarios],
                FIG_DIR / "nmse_vs_snr_grid.png")


def fig_sumrate_grid(scenarios=("v30", "v50", "v90", "v120")):
    _stitch_2x2([FIG_DIR / f"sumrate_vs_snr_{sc}.png" for sc in scenarios],
                FIG_DIR / "sumrate_vs_snr_grid.png")


# ---------------------------------------------------------------------------
# CDF của NMSE per-sample (độ tin cậy / robustness) — hình 4.x
# ---------------------------------------------------------------------------
def fig_nmse_cdf(models, normalizer, device, data_dir, scenario="v90",
                 methods=("pi_dast", "transformer", "lstm", "gru", "cnn",
                          "pad", "kalman", "no_pred"), n=None):
    """Phân bố tích lũy (CDF) NMSE TỪNG MẪU trên test_<scenario>. Cho thấy độ ổn
    định: bao nhiêu % mẫu có NMSE dưới 1 ngưỡng (không bị che bởi trung bình)."""
    samples = load_saved_samples(data_dir / f"test_{scenario}", limit=n)
    if not samples:
        print(f"[cdf] BỎ QUA: thiếu test_{scenario}"); return
    plt.figure(figsize=(8, 5)); ax = plt.gca()
    for m in methods:
        if m in MODELS and m not in models:
            continue
        pr, tr = collect_predictions(m, models, samples, normalizer, device)
        num = ((pr - tr) ** 2).reshape(pr.shape[0], -1).sum(1)        # ||err||^2 mỗi mẫu
        den = (tr ** 2).reshape(tr.shape[0], -1).sum(1) + 1e-12       # ||true||^2 mỗi mẫu
        nmse_db = 10.0 * np.log10(num / den)
        xs = np.sort(nmse_db); ys = np.arange(1, len(xs) + 1) / len(xs)
        c, ls, _, lw = STYLE[m]
        ax.plot(xs, ys, color=c, linestyle=ls, linewidth=max(lw, 1.8), label=DISP[m])
    ax.axhline(0.9, color="grey", ls=":", lw=0.8)                    # mốc 90%
    ax.set_xlabel("NMSE mỗi mẫu (dB, thấp = tốt)"); ax.set_ylabel("CDF — xác suất tích luỹ")
    ax.set_title(f"Phân bố tích luỹ NMSE — v={scenario[1:]} km/h (SNR test 20 dB)")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    p = FIG_DIR / f"nmse_cdf_{scenario}.png"; plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig] {p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_sionna")
    ap.add_argument("--exp_suffix", default="exp1_seed42")
    ap.add_argument("--n_sweep", type=int, default=200)
    ap.add_argument("--skip_sweeps", action="store_true", help="bỏ NMSE-vs-SNR/velocity")
    ap.add_argument("--skip_ber", action="store_true", help="bỏ BER")
    ap.add_argument("--skip_complexity", action="store_true")
    ap.add_argument("--skip_ablation", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")
    FIG_DIR.mkdir(parents=True, exist_ok=True); TBL_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    normalizer = ChannelNormalizer.load(data_dir / "normalize_stats.npz")
    models = load_models(args.exp_suffix, device)
    if not models:
        print("[eval] không có model -> hãy train trước. Dừng."); return

    summary = {"exp_suffix": args.exp_suffix, "models": list(models), "device": device.type}

    fig_learning_curves(args.exp_suffix)                              # PHẦN 2.1
    eval_scenarios(models, normalizer, device, data_dir,             # PHẦN 2.2 + sumrate + BER
                   do_ber=not args.skip_ber, summary=summary)
    if not args.skip_ber:
        fig_ber_grid()                                               # ghép 4 BER -> lưới 2x2
    fig_sumrate_grid()                                               # ghép 4 sum-rate -> lưới 2x2
    if not args.skip_sweeps:                                         # PHẦN 3
        fig_nmse_vs_snr(models, normalizer, device, args.n_sweep, summary, data_dir)
        fig_nmse_snr_grid()                                          # ghép 4 NMSE-vs-SNR -> lưới 2x2
        fig_nmse_vs_velocity(models, normalizer, device, args.n_sweep, summary, data_dir)
    if not args.skip_complexity:                                     # PHẦN 4
        complexity_table(models, device, summary)
    if not args.skip_ablation:                                       # PHẦN 4 (ablation)
        ablation_table(normalizer, device, data_dir, summary)
    fig_pred_vs_true(models, normalizer, device, data_dir)
    fig_doppler_gate(models, normalizer, device, data_dir)           # explainability cổng
    fig_nmse_cdf(models, normalizer, device, data_dir)               # CDF NMSE per-sample (v90)

    json.dump(summary, open(TBL_DIR / "evaluate_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"\n[eval] XONG. Hình -> {FIG_DIR}")
    print(f"[eval] Bảng -> {TBL_DIR}/complexity_table.md , evaluate_summary.json")


if __name__ == "__main__":
    main()
