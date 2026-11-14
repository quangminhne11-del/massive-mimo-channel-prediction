import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Tái dùng toàn bộ hạ tầng eval (model, data, metric) -> 1 nguồn sự thật.
from evaluate import (load_saved_samples, renoise_samples, collect_predictions,
                      nmse_per_frame_db, DISP, STYLE)
from src.models.utils import build_model, count_trainable_params, is_spatial_model
from src.utils.config import load_config
from src.data.normalize import ChannelNormalizer

# ============================================================
# CẤU HÌNH
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = ROOT / "data_sionna"
CKPT = ROOT / "results" / "checkpoints"
NORM = ChannelNormalizer.load(DATA / "normalize_stats.npz")

VELS = [30, 50, 70, 90, 120]
VEL_LABELS = {30: "30 km/h · Đô thị", 50: "50 km/h · Ngoại ô", 70: "70 km/h · Tỉnh lộ",
              90: "90 km/h · Cao tốc", 120: "120 km/h · Cao tốc nhanh"}
LBL2VEL = {v: k for k, v in VEL_LABELS.items()}

# Lineup gọn: DAST (đề xuất) + baseline Transformer/LSTM/GRU/CNN. PAD = cổ điển (tab so sánh).
APP_MODELS = ["pi_dast", "transformer", "lstm", "gru", "cnn"]
SEEDS = {"pi_dast": [42, 123, 2024], "transformer": [42, 123, 2024]}
P, Q, NRF, NM = 10, 5, 8, 4

_cache = {}

# ============================================================
# DARK THEME
# ============================================================
DARK_BG, CARD_BG, BORDER = "#0D1117", "#161B22", "#30363D"
TEXT_PRI, TEXT_SEC = "#F0F6FC", "#8B949E"
TRUE_C = "#2ECC71"
plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG, "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT_PRI, "xtick.color": TEXT_SEC, "ytick.color": TEXT_SEC,
    "text.color": TEXT_PRI, "grid.color": BORDER, "grid.alpha": 0.4,
    "legend.facecolor": CARD_BG, "legend.edgecolor": BORDER,
    "font.family": "DejaVu Sans",  # đủ glyph tiếng Việt (monospace thiếu dấu -> ô vuông)
})


# ============================================================
# MODEL / DATA HELPERS
# ============================================================
def get_model(name, seed=42):
    seeds = SEEDS.get(name, [42])
    seed = seed if seed in seeds else seeds[0]
    key = (name, seed)
    if key in _cache:
        return _cache[key]
    exp = f"{name}_exp1_seed{seed}"
    ck = CKPT / exp / "checkpoint_best.pt"
    if not ck.exists():
        _cache[key] = None
        return None
    cfg = load_config(str(ROOT / "configs" / "model" / f"{name}.yaml"))
    model = build_model(name, cfg)
    model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=False)["model_state_dict"])
    model.to(DEVICE).eval()
    _cache[key] = (model, not is_spatial_model(name), count_trainable_params(model))
    return _cache[key]


def _samples(vel, n=None):
    return load_saved_samples(DATA / f"test_v{vel}", limit=n)


def _cplx(arr, rf, ue):
    """arr (...,NRF,NM,2) -> complex (...) tại (rf,ue)."""
    return arr[..., rf, ue, 0] + 1j * arr[..., rf, ue, 1]


def _predict(name, sample_noisy, seed=42):
    if name in ("pad", "kalman", "no_pred"):             # cổ điển / tham chiếu: không cần checkpoint
        pr, _ = collect_predictions(name, {}, [sample_noisy], NORM, DEVICE)
        return pr[0]
    md = get_model(name, seed)
    if md is None:
        return None
    pr, _ = collect_predictions(name, {name: md}, [sample_noisy], NORM, DEVICE)
    return pr[0]  # (Q,NRF,NM,2)


def _disp(name):
    return DISP.get(name, name)


def _color(name):
    return STYLE.get(name, ("#888888", "-", "o", 1.5))[0]


# ============================================================
# TAB 1: DỰ ĐOÁN KÊNH (|He| theo thời gian, đa model)
# ============================================================
def tab_predict(vel_label, seed, snr, sample_idx, rf, ue, selected):
    vel = LBL2VEL[vel_label]
    seed, rf, ue, sample_idx = int(seed), int(rf), int(ue), int(sample_idx)
    sm = _samples(vel)
    if not sm:
        return None, "❌ Không có data test_v%d" % vel
    if not selected:
        return None, "⚠️ Chọn ít nhất 1 model"
    sample_idx = min(sample_idx, len(sm) - 1)
    orig = sm[sample_idx]
    noisy = renoise_samples([orig], snr, seed=sample_idx)[0]

    hist_c = _cplx(orig["He_history_clean"], rf, ue)     # (P,)
    fut_t = _cplx(orig["He_future"], rf, ue)             # (Q,)
    t_hist = np.arange(P)
    t_fut = np.arange(P, P + Q)

    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.8 * n + 0.6), squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor(DARK_BG)

    rows = [f"📡 v={vel} km/h · SNR={snr:.0f} dB · seed={seed} · mẫu #{sample_idx} · He[RF{rf},UE{ue}]\n",
            f"{'Model':<26}{'NMSE (dB)':>19}"]
    rows.append("-" * 44)

    for ax, name in zip(axes, selected):
        pr = _predict(name, noisy, seed)
        ax.axvspan(0, P - 1, color="#1f2630", alpha=0.5, zorder=0)
        ax.plot(t_hist, np.abs(hist_c), color=TEXT_SEC, lw=1.4, marker="o", ms=3,
                label="Lịch sử (sạch)", zorder=3)
        ax.plot(t_fut, np.abs(fut_t), color=TRUE_C, lw=2.0, marker="o", ms=4,
                label="Tương lai THẬT", zorder=4)
        # nối lịch sử -> tương lai cho liền mạch
        ax.plot([P - 1, P], [np.abs(hist_c[-1]), np.abs(fut_t[0])], color=TRUE_C, lw=1.0, alpha=0.5, zorder=3)
        if pr is not None:
            fp = _cplx(pr, rf, ue)
            c = _color(name)
            ax.plot(t_fut, np.abs(fp), color=c, lw=2.0, ls="--", marker="s", ms=4,
                    label=f"{_disp(name)} (dự đoán)", zorder=5)
            ax.fill_between(t_fut, np.abs(fut_t), np.abs(fp), color=c, alpha=0.25,
                            label="Sai số", zorder=2)
            nm_db = nmse_per_frame_db(pr[None], orig["He_future"][None])[1]
            rows.append(f"{_disp(name):<26}{nm_db:>16.2f} dB")
            ax.set_title(f"{_disp(name)}   ·   NMSE = {nm_db:.2f} dB", color=c, fontsize=11, pad=6)
        else:
            ax.text(0.5, 0.5, f"Thiếu checkpoint: {name}", ha="center", va="center",
                    color=TEXT_SEC, transform=ax.transAxes)
        ax.axvline(P - 0.5, color="#888", ls=":", lw=1)
        ax.set_ylabel("|He(t)|", fontsize=9)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=2)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Khung thời gian  (0..P lịch sử | P..P+Q tương lai, x0.125 ms)", fontsize=10)
    fig.tight_layout()
    return fig, "\n".join(rows)


# ============================================================
# TAB 2: TÍN HIỆU CHI TIẾT (Re/Im + quỹ đạo phức)
# ============================================================
def tab_signal(vel_label, seed, snr, sample_idx, rf, ue, name):
    vel = LBL2VEL[vel_label]
    seed, rf, ue, sample_idx = int(seed), int(rf), int(ue), int(sample_idx)
    sm = _samples(vel)
    if not sm:
        return None, "❌ Không có data"
    sample_idx = min(sample_idx, len(sm) - 1)
    orig = sm[sample_idx]
    noisy = renoise_samples([orig], snr, seed=sample_idx)[0]
    pr = _predict(name, noisy, seed)
    if pr is None:
        return None, f"❌ Thiếu checkpoint {name}"

    hist_c = _cplx(orig["He_history_clean"], rf, ue)
    fut_t = _cplx(orig["He_future"], rf, ue)
    fp = _cplx(pr, rf, ue)
    t_hist, t_fut = np.arange(P), np.arange(P, P + Q)
    c = _color(name)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    fig.patch.set_facecolor(DARK_BG)

    for ax, part, fn in ((axes[0], "Thực (Real)", np.real), (axes[1], "Ảo (Imag)", np.imag)):
        ax.axvspan(0, P - 1, color="#1f2630", alpha=0.5, zorder=0)
        ax.plot(t_hist, fn(hist_c), color=TEXT_SEC, lw=1.4, marker="o", ms=3, label="Lịch sử")
        ax.plot(t_fut, fn(fut_t), color=TRUE_C, lw=2.0, marker="o", ms=4, label="THẬT")
        ax.plot(t_fut, fn(fp), color=c, lw=2.0, ls="--", marker="s", ms=4, label="Dự đoán")
        ax.fill_between(t_fut, fn(fut_t), fn(fp), color=c, alpha=0.25)
        ax.axvline(P - 0.5, color="#888", ls=":", lw=1)
        ax.set_title(f"Thành phần {part}", color=TEXT_PRI, fontsize=11)
        ax.set_xlabel("Khung thời gian", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.85); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"He[RF{rf},UE{ue}]", fontsize=10)

    # Quỹ đạo phức
    ax = axes[2]
    ax.plot(np.real(hist_c), np.imag(hist_c), color=TEXT_SEC, lw=1.2, marker="o", ms=3, label="Lịch sử")
    ax.plot(np.real(fut_t), np.imag(fut_t), color=TRUE_C, lw=2.0, marker="o", ms=5, label="THẬT")
    ax.plot(np.real(fp), np.imag(fp), color=c, lw=2.0, ls="--", marker="s", ms=5, label="Dự đoán")
    for a, b in zip(fut_t, fp):  # nối lệch true-pred từng frame
        ax.plot([a.real, b.real], [a.imag, b.imag], color="#E74C3C", lw=0.8, alpha=0.6)
    ax.set_title("Quỹ đạo mặt phẳng phức", color=TEXT_PRI, fontsize=11)
    ax.set_xlabel("Re"); ax.set_ylabel("Im"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()

    nm_db = nmse_per_frame_db(pr[None], orig["He_future"][None])[1]
    pf = np.array(nmse_per_frame_db(pr[None], orig["He_future"][None])[0])
    txt = (f"**{_disp(name)}** · v={vel} · SNR={snr:.0f}dB · He[RF{rf},UE{ue}]\n\n"
           f"NMSE tổng: **{nm_db:.2f} dB**\n\n"
           f"NMSE từng khung tương lai (dB): " + " · ".join(f"f{i+1}={x:.1f}" for i, x in enumerate(pf)))
    return fig, txt


# ============================================================
# TAB 3: SO SÁNH NMSE
# ============================================================
def tab_compare(vel_label, n_eval):
    vel = LBL2VEL[vel_label]
    n_eval = int(n_eval)
    sm = _samples(vel, n_eval)
    if not sm:
        return None
    methods = APP_MODELS + ["pad", "kalman", "no_pred"]
    vals = {}
    for m in methods:
        if m in APP_MODELS and get_model(m) is None:
            continue
        md = {m: get_model(m)} if m in APP_MODELS else {}
        pr, tr = collect_predictions(m, md, sm, NORM, DEVICE)
        vals[m] = nmse_per_frame_db(pr, tr)[1]            # dB
    order = sorted(vals, key=vals.get)

    fig, ax = plt.subplots(figsize=(11, 5.5)); fig.patch.set_facecolor(DARK_BG)
    bars = ax.bar(range(len(order)), [vals[m] for m in order],
                  color=[_color(m) for m in order], alpha=0.9)
    ax.axhline(0.0, color="#E74C3C", ls=":", lw=1.2, label="0 dB = 'không dự đoán'")
    for b, m in zip(bars, order):
        off = 0.15 if vals[m] >= 0 else -0.4
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + off,
                f"{vals[m]:.1f}", ha="center", fontsize=8, color=TEXT_PRI)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_disp(m) for m in order], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("NMSE (dB, thấp = tốt)", fontsize=11)
    ax.set_title(f"So sánh NMSE · v={vel} km/h · {len(sm)} mẫu · SNR=20 dB", color=TEXT_PRI, fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def tab_velocity_curve(n_eval):
    n_eval = int(n_eval)
    show = ["pi_dast", "transformer", "pad", "kalman"]
    fig, ax = plt.subplots(figsize=(10, 5.5)); fig.patch.set_facecolor(DARK_BG)
    for m in show:
        ys = []
        for v in VELS:
            sm = _samples(v, n_eval)
            md = {m: get_model(m)} if m in APP_MODELS else {}
            if m in APP_MODELS and get_model(m) is None:
                ys.append(np.nan); continue
            pr, tr = collect_predictions(m, md, sm, NORM, DEVICE)
            ys.append(nmse_per_frame_db(pr, tr)[1])       # dB
        c, ls, mk, lw = STYLE.get(m, ("#888", "-", "o", 1.5))
        ax.plot(VELS, ys, color=c, ls=ls, marker=mk, lw=2.2 if m == "pi_dast" else 1.6,
                ms=7 if m == "pi_dast" else 5, label=_disp(m))
    ax.axhline(0.0, color="#E74C3C", ls=":", lw=1.2)
    ax.set_xlabel("Vận tốc (km/h)", fontsize=11); ax.set_ylabel("NMSE (dB, thấp = tốt)", fontsize=11)
    ax.set_title(f"NMSE theo vận tốc · {n_eval} mẫu/điểm · SNR=20 dB", color=TEXT_PRI, fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# TAB 4: THÔNG SỐ MODEL
# ============================================================
def tab_stats():
    rows, names, params = [], [], []
    for name in APP_MODELS:
        cfg = load_config(str(ROOT / "configs" / "model" / f"{name}.yaml"))
        model = build_model(name, cfg).to(DEVICE).eval()
        p = count_trainable_params(model)
        flatten = not is_spatial_model(name)
        dummy = (torch.randn(1, P, NRF, NM, 2, device=DEVICE) if not flatten
                 else torch.randn(1, P, NRF * NM * 2, device=DEVICE))
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                model(dummy)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) / 30 * 1000
        names.append(_disp(name)); params.append(p)
        rows.append((name, p, lat))

    fig, ax = plt.subplots(figsize=(11, 5)); fig.patch.set_facecolor(DARK_BG)
    bars = ax.barh(names, params, color=[_color(n) for n, _, _ in rows], alpha=0.9, height=0.6)
    for b, p in zip(bars, params):
        ax.text(p + max(params) * 0.01, b.get_y() + b.get_height() / 2, f"{p:,}",
                va="center", fontsize=9, color=TEXT_PRI)
    ax.set_xlabel("Số tham số", fontsize=11); ax.set_title("Độ phức tạp model", color=TEXT_PRI, fontsize=12)
    ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    md = "| Model | Tham số | Latency (ms, batch=1) |\n|---|---|---|\n"
    for name, p, lat in rows:
        star = " ⭐" if name == "pi_dast" else ""
        md += f"| **{_disp(name)}**{star} | {p:,} | {lat:.2f} |\n"
    return fig, md


# ============================================================
# CSS + HEADER (giữ phong cách dark cũ)
# ============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap');
:root{--bg:#0D1117;--card:#161B22;--border:#30363D;--txt:#F0F6FC;--txt2:#8B949E;--ac:#9B59B6;--ac2:#3498DB;}
body,.gradio-container{background:var(--bg)!important;font-family:'Space Grotesk',sans-serif!important;color:var(--txt)!important;}
.gr-panel,.gr-box,.gr-block{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;}
.gr-button{background:linear-gradient(135deg,var(--ac),var(--ac2))!important;border:none!important;color:#fff!important;font-weight:600!important;border-radius:8px!important;padding:10px 24px!important;}
.gr-button:hover{opacity:.88!important;box-shadow:0 4px 20px rgba(155,89,182,.4)!important;}
.gr-tab-nav button.selected{color:var(--ac)!important;border-bottom:2px solid var(--ac)!important;}
code,.gr-textbox textarea{font-family:'JetBrains Mono',monospace!important;background:#0D1117!important;color:#79C0FF!important;}
#hdr{background:linear-gradient(135deg,#0D1117 0%,#1a0a2e 50%,#0D1117 100%);border:1px solid var(--border);border-radius:16px;padding:24px 32px;margin-bottom:14px;}
"""
HEADER = """
<div id="hdr"><div style="display:flex;align-items:center;gap:16px;">
<div style="font-size:44px;">📡</div>
<div><h1 style="margin:0;font-size:22px;font-weight:700;background:linear-gradient(90deg,#9B59B6,#3498DB);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
DAST · Channel Prediction Dashboard</h1>
<p style="margin:4px 0 0;color:#8B949E;font-size:13px;font-family:monospace;">
Sionna CDL-A · MIMO 8×4 · P=10 → Q=5 · Doppler-Adaptive Sequence Transformer</p></div>
<div style="margin-left:auto;"><span style="background:#1a2a1a;color:#2ECC71;padding:4px 12px;border-radius:20px;font-size:12px;font-family:monospace;">● Sẵn sàng</span></div>
</div></div>
"""


def build_app():
    avail = [m for m in APP_MODELS if get_model(m) is not None]
    with gr.Blocks(css=CSS, title="Channel Prediction", theme=gr.themes.Base()) as demo:
        gr.HTML(HEADER)
        with gr.Tabs():
            # TAB 1
            with gr.TabItem("📈 Dự đoán kênh"):
                with gr.Row():
                    with gr.Column(scale=1):
                        v1 = gr.Dropdown([VEL_LABELS[v] for v in VELS], value=VEL_LABELS[30], label="🚀 Vận tốc")
                        sd1 = gr.Dropdown(["42", "123", "2024"], value="42", label="🎲 Seed (pi_dast/transformer)")
                        snr1 = gr.Slider(0, 30, value=20, step=5, label="📶 SNR (dB)")
                        idx1 = gr.Slider(0, 499, value=0, step=1, label="🔢 Mẫu test")
                        rf1 = gr.Slider(0, NRF - 1, value=0, step=1, label="📡 RF chain")
                        ue1 = gr.Slider(0, NM - 1, value=0, step=1, label="👤 UE antenna")
                        md1 = gr.CheckboxGroup([(_disp(m), m) for m in APP_MODELS + ["pad", "kalman"]],
                                               value=["pi_dast", "transformer", "pad"], label="🤖 Model (PAD = cổ điển)")
                        b1 = gr.Button("▶ Dự đoán", variant="primary")
                    with gr.Column(scale=3):
                        p1 = gr.Plot(label="Dự đoán |He(t)|")
                        s1 = gr.Markdown()
                b1.click(tab_predict, [v1, sd1, snr1, idx1, rf1, ue1, md1], [p1, s1])

            # TAB 2
            with gr.TabItem("🔬 Tín hiệu chi tiết"):
                with gr.Row():
                    with gr.Column(scale=1):
                        v2 = gr.Dropdown([VEL_LABELS[v] for v in VELS], value=VEL_LABELS[30], label="🚀 Vận tốc")
                        sd2 = gr.Dropdown(["42", "123", "2024"], value="42", label="🎲 Seed")
                        snr2 = gr.Slider(0, 30, value=20, step=5, label="📶 SNR (dB)")
                        idx2 = gr.Slider(0, 499, value=0, step=1, label="🔢 Mẫu test")
                        rf2 = gr.Slider(0, NRF - 1, value=0, step=1, label="📡 RF chain")
                        ue2 = gr.Slider(0, NM - 1, value=0, step=1, label="👤 UE antenna")
                        m2 = gr.Dropdown([(_disp(m), m) for m in APP_MODELS + ["pad", "kalman"]], value="pi_dast", label="🤖 Model")
                        b2 = gr.Button("▶ Xem tín hiệu", variant="primary")
                    with gr.Column(scale=3):
                        p2 = gr.Plot(label="Re / Im / Quỹ đạo phức")
                        s2 = gr.Markdown()
                b2.click(tab_signal, [v2, sd2, snr2, idx2, rf2, ue2, m2], [p2, s2])

            # TAB 3
            with gr.TabItem("📊 So sánh NMSE"):
                with gr.Row():
                    with gr.Column(scale=1):
                        v3 = gr.Dropdown([VEL_LABELS[v] for v in VELS], value=VEL_LABELS[30], label="🚀 Vận tốc")
                        n3 = gr.Slider(50, 500, value=200, step=50, label="📦 Số mẫu đánh giá")
                        b3 = gr.Button("▶ So sánh model", variant="primary")
                        b3v = gr.Button("▶ NMSE theo vận tốc", variant="primary")
                    with gr.Column(scale=3):
                        p3 = gr.Plot(label="NMSE theo model")
                        p3v = gr.Plot(label="NMSE theo vận tốc")
                b3.click(tab_compare, [v3, n3], [p3])
                b3v.click(tab_velocity_curve, [n3], [p3v])

            # TAB 4
            with gr.TabItem("⚙️ Thông số model"):
                b4 = gr.Button("▶ Tải thông số", variant="primary")
                p4 = gr.Plot(label="Tham số")
                t4 = gr.Markdown()
                b4.click(tab_stats, [], [p4, t4])

        gr.HTML('<div style="text-align:center;padding:18px;color:#4A5568;font-family:monospace;font-size:12px;">'
                'DAST · Doppler-Adaptive Sequence Transformer · HCMUTE 2026</div>')
    return demo


if __name__ == "__main__":
    print("=" * 56)
    print("  DAST Channel Prediction Dashboard")
    print(f"  Device: {DEVICE}")
    avail = [m for m in APP_MODELS if get_model(m) is not None]
    print(f"  Checkpoint sẵn: {avail}")
    print("=" * 56)
    build_app().launch(server_port=7860, show_error=True, inbrowser=True)
