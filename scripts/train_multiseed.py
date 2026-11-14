# Multi-seed: train pi_dast / transformer / nodoppler ở seed 123 & 2024
# (seed 42 đã có). Tất cả CÙNG recipe exp1 (NMSE-only) -> so sánh + ablation
# Doppler single-variable, báo mean±std qua 3 seed.
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG_DIR = ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [123, 2024]
# (model, train_config, model_config_override, exp_name_tmpl)
SPECS = [
    ("pi_dast", "exp1", None, "pi_dast_exp1_seed{s}"),
    ("pi_dast", "exp1", "configs/model/pi_dast_nodoppler.yaml", "pi_dast_nodoppler_seed{s}"),
    ("transformer", "exp1", None, "transformer_exp1_seed{s}"),
]


def main():
    runs = [(m, tc, mc, name.format(s=s), s) for s in SEEDS for (m, tc, mc, name) in SPECS]
    results = []
    for i, (model, tcfg, mcfg, exp, seed) in enumerate(runs, 1):
        cmd = [PY, str(ROOT / "scripts" / "02_train.py"),
               "--model", model, "--train_config", tcfg, "--data_dir", "data_sionna",
               "--seed", str(seed), "--exp_name", exp]
        if mcfg:
            cmd += ["--model_config", mcfg]
        log_path = LOG_DIR / f"driver_{exp}.log"
        print(f"[{i}/{len(runs)}] {exp} -> {log_path.name}", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        dt = (time.time() - t0) / 60
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[{i}/{len(runs)}] {exp}: {status} ({dt:.1f} min)", flush=True)
        results.append((exp, status, dt))
    print("\n===== SUMMARY =====", flush=True)
    for exp, status, dt in results:
        print(f"{exp:35s} {status:12s} {dt:7.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
