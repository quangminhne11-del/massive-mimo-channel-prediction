import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from evaluate import (collect_predictions, nmse_per_frame_db,
                      load_saved_samples, nmse_lin)
from src.models.utils import build_model, count_trainable_params, is_spatial_model
from src.utils.config import load_config
from src.data.normalize import ChannelNormalizer

SEEDS = [42, 123, 2024]
DATA = ROOT / "data_sionna"
SCEN = ["v30", "v50", "v90", "v120"]
# (nhãn, model_key, exp_name_tmpl, model_cfg)
SPECS = [
    ("DAST (hybrid, đề xuất)", "pi_dast",     "pi_dast_exp1_seed{s}",      "configs/model/pi_dast.yaml"),
    ("DAST (w/o gate)",        "pi_dast",     "pi_dast_fixedgate_seed{s}", "configs/model/pi_dast_fixedgate.yaml"),
    ("DAST (w/o PAD-prior)",   "pi_dast",     "pi_dast_nopad_seed{s}",     "configs/model/pi_dast_nopad.yaml"),
    ("DAST (w/o Doppler-enc)", "pi_dast",     "pi_dast_nodoppler_seed{s}", "configs/model/pi_dast_nodoppler.yaml"),
    ("Transformer",            "transformer", "transformer_exp1_seed{s}",  "configs/model/transformer.yaml"),
]


def require_complete_checkpoints():
    # Fail before aggregation so the output cannot claim three seeds when one is missing.
    missing = []
    for label, _model_key, exp_template, _cfg_path in SPECS:
        for seed in SEEDS:
            checkpoint = ROOT / "results" / "checkpoints" / exp_template.format(s=seed) / "checkpoint_best.pt"
            if not checkpoint.is_file():
                missing.append(f"{label}: seed {seed} ({checkpoint.relative_to(ROOT)})")
    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise SystemExit(
            "Cannot aggregate the multi-seed table; required checkpoints are missing:\n"
            f"{details}\n"
            "Provide every required model/seed checkpoint, or explicitly update the seed/model list "
            "in this script before generating a table with a different label."
        )


def main():
    require_complete_checkpoints()
    norm = ChannelNormalizer.load(DATA / "normalize_stats.npz")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = {sc: load_saved_samples(DATA / f"test_{sc}") for sc in SCEN}

    table = {}  # label -> {scen: [nmse_lin per seed], 'ber': [...]}
    for label, mkey, tmpl, cfgp in SPECS:
        cfg = load_config(str(ROOT / cfgp))
        per = {sc: [] for sc in SCEN}
        used = []
        for s in SEEDS:
            ck = ROOT / "results" / "checkpoints" / tmpl.format(s=s) / "checkpoint_best.pt"
            if not ck.is_file():
                raise RuntimeError(
                    f"Checkpoint disappeared during aggregation: {label}, seed {s}: {ck}"
                )
            model = build_model(mkey, cfg)
            model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model_state_dict"])
            model.to(dev).eval()
            tmp = {mkey: (model, not is_spatial_model(mkey), count_trainable_params(model))}
            for sc in SCEN:
                pr, tr = collect_predictions(mkey, tmp, samples[sc], norm, dev)
                per[sc].append(nmse_per_frame_db(pr, tr)[1])     # dB
            used.append(s)
        if used != SEEDS:
            raise RuntimeError(
                f"Incomplete seeds for {label}: expected {SEEDS}, found {used}; "
                "multi-seed table was not written."
            )
        table[label] = {"per": per, "seeds": used}
        msg = " ".join(f"{sc}={np.mean(per[sc]):.2f}±{np.std(per[sc]):.2f}dB" for sc in SCEN)
        print(f"[agg] {label:24s} seeds={used} {msg}")

    lines = ["# Multi-seed (3 seeds: 42/123/2024) — mean ± std\n",
             "NMSE dạng dB (thấp = tốt). BER/Sum-rate xem hình waterfall riêng.\n",
             "| Model | NMSE v30 (dB) | NMSE v50 (dB) | NMSE v90 (dB) | NMSE v120 (dB) |",
             "|---|---|---|---|---|"]
    for label, mkey, tmpl, cfgp in SPECS:
        d = table[label]
        cells = " | ".join(f"{np.mean(d['per'][sc]):.2f} ± {np.std(d['per'][sc]):.2f}" for sc in SCEN)
        lines.append(f"| {label} | {cells} |")
    out = ROOT / "results" / "tables" / "multiseed_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[tbl] {out}")


if __name__ == "__main__":
    main()
