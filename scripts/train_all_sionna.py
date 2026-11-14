# Driver: train tuần tự các run trên data_sionna (DAST + baselines + ablation Doppler).
# Lineup cuối: pi_dast (DAST), transformer, cnn, gru, lstm, rnn. (st_mamba/informer/
# hybrid_attention đã loại + xoá khỏi repo.) Cổ điển PAD/Kalman không train.
# Mỗi run log riêng results/logs/driver_<exp>.log; lỗi 1 run không chặn run sau.
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG_DIR = ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# (model, train_config, model_config_override, exp_name)
RUNS = [
    ("pi_dast", "exp1_phys", None, "pi_dast_exp1_seed42"),
    ("pi_dast", "exp1_phys", "configs/model/pi_dast_nodoppler.yaml", "pi_dast_nodoppler_seed42"),
    ("pi_dast", "exp1", None, "pi_dast_nophys_seed42"),
    ("transformer", "exp1", None, "transformer_exp1_seed42"),
    ("cnn", "exp1", None, "cnn_exp1_seed42"),
    ("gru", "exp1", None, "gru_exp1_seed42"),
    ("lstm", "exp1", None, "lstm_exp1_seed42"),
    ("rnn", "exp1", None, "rnn_exp1_seed42"),
]


def main():
    results = []
    for i, (model, tcfg, mcfg, exp) in enumerate(RUNS, 1):
        cmd = [
            PY, str(ROOT / "scripts" / "02_train.py"),
            "--model", model,
            "--train_config", tcfg,
            "--data_dir", "data_sionna",
            "--seed", "42",
            "--exp_name", exp,
        ]
        if mcfg:
            cmd += ["--model_config", mcfg]
        log_path = LOG_DIR / f"driver_{exp}.log"
        print(f"[{i}/{len(RUNS)}] {exp} -> {log_path.name}", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        dt = (time.time() - t0) / 60
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[{i}/{len(RUNS)}] {exp}: {status} ({dt:.1f} min)", flush=True)
        results.append((exp, status, dt))

    print("\n===== SUMMARY =====", flush=True)
    for exp, status, dt in results:
        print(f"{exp:35s} {status:12s} {dt:7.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
