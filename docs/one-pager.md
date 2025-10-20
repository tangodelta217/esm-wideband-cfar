# esm-wideband-cfar — One Pager

## Objetivo
- EN: Deliver a reproducible wideband Cell-Averaging CFAR reference for EW/SIGINT prototyping with Python + GNU Radio, SigMF interoperability, and automated Pd/Pfa validation.
- ES: Proveer un referente reproducible de CFAR de banda ancha para prototipado EW/SIGINT con Python + GNU Radio, interoperable con SigMF y con validación automatizada de Pd/Pfa.

## Arquitectura
- Signal ingest (SigMF or live SDR) → FFT/vectorization → `ca_cfar_1d_block` → detections + tracking.
- Python packages (`src/cfar`, `src/gr_blocks`) isolate algorithms from GNU Radio glue.
- Evaluation stack: `eval/metrics.py` Monte Carlo engine + `eval/evaluate_cfar.py` CLI → JSON summaries → plotting/notebooks (user supplied).
- Tooling: Make targets (`make demo|eval|lint`), GitHub Actions (tests + Ruff), VSCode settings, IQEngine integration.

## KPIs
- Probability of detection (Pd) vs. SNR bins.
- Empirical probability of false alarm (Pfa) and false alarms per MHz.
- Processing latency (median, 95.º) from IQ sample ingestion to detection output.
- Track continuity (drop rate, average dwell) using `PeakTracker`.

## Figuras sugeridas
- Pd vs. SNR curve (Monte Carlo results in `results/pd_pfa_summary*.json`).
- Pfa vs. threshold tuning sweep (`pfa` knob, training/guard cells).
- Spectrum + CFAR threshold overlay (GNU Radio vector sinks or IQEngine waterfall).
- Track evolution over time (frequency vs. frame index).

## Lecciones aprendidas
- Adaptive thresholding remains sensitive to window sizing; calibrate training/guard cells per scenario.
- Empirical Pfa validation is fast with synthetic data and prevents overconfidence in analytic α.
- SigMF + IQEngine provide quick human validation; ensure metadata (fs, fc) is accurate to avoid misalignment.
- Latency budgeting matters: Python prototypes are adequate for algorithm tuning; migrate bottlenecks to C++/FPGA for deployment.
- Automate everything: `make setup/test/eval` keeps the lab repeatable and shortens onboarding for new analysts.

