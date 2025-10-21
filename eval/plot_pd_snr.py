from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]

try:
    from eval.pd_tools import sweep_snr
except ImportError:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from eval.pd_tools import sweep_snr

matplotlib.use("Agg")  # evita problemas de display en servidores

import matplotlib.pyplot as plt  # noqa: E402

snrs = list(range(-10, 21, 2))
results = sweep_snr(snrs)
prob_detection = [row[1] for row in results]
pfa = results[0][2]

plt.figure(figsize=(6, 4))
plt.plot(snrs, prob_detection, marker="o")
plt.xlabel("SNR [dB]")
plt.ylabel("Pd")
plt.title(f"Pd vs SNR (pfa≈{pfa:.1e})")
plt.grid(True)
plt.tight_layout()

pathlib.Path("docs").mkdir(exist_ok=True)
plt.savefig("docs/pd_snr_curve.png", dpi=200)
print("Figura lista en docs/pd_snr_curve.png")
