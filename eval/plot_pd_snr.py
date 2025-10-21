import matplotlib
matplotlib.use("Agg")  # evita problemas de display en servidores
import matplotlib.pyplot as plt
from eval.pd_tools import sweep_snr

snrs = list(range(-10, 21, 2))
res  = sweep_snr(snrs)
pd   = [r[1] for r in res]
pfa  = res[0][2]

plt.figure(figsize=(6,4))
plt.plot(snrs, pd, marker='o')
plt.xlabel("SNR [dB]"); plt.ylabel("Pd"); plt.title(f"Pd vs SNR (pfa≈{pfa:.1e})")
plt.grid(True); plt.tight_layout()

import pathlib
pathlib.Path("docs").mkdir(exist_ok=True)
plt.savefig("docs/pd_snr_curve.png", dpi=200)
print("Figura lista en docs/pd_snr_curve.png")
