# Visualize data_sionna trước khi train: time-evolution, complex trajectory,
# Doppler spectrum, autocorrelation, phân bố metadata train.
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_sionna"
OUT = ROOT / "results" / "figures_data"
OUT.mkdir(parents=True, exist_ok=True)

T_SAMPLE_MS = 0.625
VELS = [30, 50, 70, 90, 120]


def load_complex(path):
    d = np.load(path)
    h = np.concatenate([d["He_history"], d["He_future"]], axis=0)  # (35,8,4,2)
    return h[..., 0] + 1j * h[..., 1]  # (35,8,4)


def fig1_time_evolution():
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    t = np.arange(35) * T_SAMPLE_MS
    for ax, v in zip(axes.flat, VELS):
        h = load_complex(DATA / f"test_v{v}" / "sample_00000.npz")
        for rf, m in [(0, 0), (3, 1), (6, 3)]:
            ax.plot(t, np.abs(h[:, rf, m]), label=f"RF{rf},UE{m}")
        ax.axvline(30 * T_SAMPLE_MS, color="k", ls="--", lw=1)
        ax.set_title(f"v = {v} km/h")
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|H_e|$")
    fig.suptitle("Sionna CDL-A: |He| theo thời gian (nét đứt = ranh history|future, P=30/Q=5)")
    fig.tight_layout()
    fig.savefig(OUT / "data_time_evolution.png", dpi=150)
    plt.close(fig)


def fig2_complex_traj():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, v in zip(axes, [30, 70, 120]):
        h = load_complex(DATA / f"test_v{v}" / "sample_00000.npz")[:, 0, 0]
        ax.plot(h.real, h.imag, ".-", ms=4)
        ax.plot(h.real[0], h.imag[0], "go", label="t=0")
        ax.plot(h.real[-1], h.imag[-1], "rs", label="t=35")
        ax.set_title(f"v = {v} km/h")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.axis("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Quỹ đạo phức He[RF0,UE0]: xoay Doppler nhanh dần theo vận tốc")
    fig.tight_layout()
    fig.savefig(OUT / "data_complex_trajectory.png", dpi=150)
    plt.close(fig)


def fig3_doppler_spectrum():
    fig, ax = plt.subplots(figsize=(8, 5))
    freqs = np.fft.fftshift(np.fft.fftfreq(30, d=T_SAMPLE_MS / 1000.0))
    for v in VELS:
        files = sorted(glob.glob(str(DATA / f"test_v{v}" / "*.npz")))[:50]
        acc = np.zeros(len(freqs))
        for f in files:
            h = load_complex(f)[:30]  # history only (input model nhìn thấy)
            spec = np.fft.fftshift(np.abs(np.fft.fft(h, axis=0)) ** 2, axes=0)
            acc += spec.mean(axis=(1, 2))
        acc /= acc.sum()
        ax.semilogy(freqs, acc, marker="o", ms=3, label=f"v={v} km/h (fD≈{v/3.6*30e9/3e8:.0f} Hz)")
    ax.set_xlabel("Tần số Doppler (Hz)")
    ax.set_ylabel("PSD chuẩn hoá")
    ax.set_title("Phổ Doppler 2 phía (50 mẫu/vận tốc, FFT theo trục thời gian P=30)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "data_doppler_spectrum.png", dpi=150)
    plt.close(fig)


def fig4_autocorr():
    fig, ax = plt.subplots(figsize=(8, 5))
    lags = np.arange(0, 20)
    for v in VELS:
        files = sorted(glob.glob(str(DATA / f"test_v{v}" / "*.npz")))[:100]
        rho = np.zeros(len(lags))
        n = 0
        for f in files:
            h = load_complex(f).reshape(35, -1)  # (35,32)
            h = h - h.mean(axis=0, keepdims=True)
            p = (np.abs(h) ** 2).mean()
            for i, L in enumerate(lags):
                rho[i] += np.real(np.conj(h[: 35 - L]) * h[L:]).mean() / p
            n += 1
        ax.plot(lags * T_SAMPLE_MS, rho / n, marker="o", ms=3, label=f"v={v} km/h")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel(r"Tự tương quan $\rho(\Delta t)$")
    ax.set_title("Tốc độ mất tương quan kênh theo vận tốc (100 mẫu/điểm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "data_autocorrelation.png", dpi=150)
    plt.close(fig)


def fig5_train_meta():
    files = sorted(glob.glob(str(DATA / "train" / "*.npz")))
    vel, ds, snr, pwr = [], [], [], []
    for f in files[::4]:  # 2000 mẫu đủ mịn
        d = np.load(f)
        vel.append(float(d["velocity"]))
        ds.append(float(d["delay_spread"]))
        snr.append(float(d["snr"]))
        pwr.append(float((d["He_history"] ** 2).sum(axis=-1).mean()))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, x, name in zip(
        axes,
        [vel, ds, snr, 10 * np.log10(np.array(pwr))],
        ["Velocity (km/h)", "Delay spread (ns)", "SNR (dB)", "Công suất He (dB)"],
    ):
        ax.hist(x, bins=40, color="steelblue", edgecolor="k", lw=0.3)
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Số mẫu")
    fig.suptitle(f"Phân bố train set ({len(vel)}/{len(files)} mẫu khảo sát)")
    fig.tight_layout()
    fig.savefig(OUT / "data_train_distributions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1_time_evolution(); print("[1/5] time_evolution")
    fig2_complex_traj(); print("[2/5] complex_trajectory")
    fig3_doppler_spectrum(); print("[3/5] doppler_spectrum")
    fig4_autocorr(); print("[4/5] autocorrelation")
    fig5_train_meta(); print("[5/5] train_distributions")
    print(f"OK -> {OUT}")
