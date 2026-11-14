# Multi-seed (3 seeds: 42/123/2024) — mean ± std

NMSE dạng dB (thấp = tốt). BER/Sum-rate xem hình waterfall riêng.

| Model | NMSE v30 (dB) | NMSE v50 (dB) | NMSE v90 (dB) | NMSE v120 (dB) |
|---|---|---|---|---|
| DAST (hybrid, đề xuất) | -7.17 ± 0.07 | -5.39 ± 0.07 | -3.37 ± 0.03 | -2.12 ± 0.02 |
| DAST (w/o gate) | -6.61 ± 0.06 | -5.14 ± 0.03 | -3.35 ± 0.03 | -2.24 ± 0.03 |
| DAST (w/o PAD-prior) | -5.62 ± 0.03 | -4.13 ± 0.05 | -2.47 ± 0.02 | -1.40 ± 0.02 |
| DAST (w/o Doppler-enc) | -5.50 ± 0.03 | -4.10 ± 0.03 | -2.43 ± 0.01 | -1.43 ± 0.04 |
| Transformer | -5.47 ± 0.07 | -4.09 ± 0.04 | -2.41 ± 0.04 | -1.38 ± 0.06 |