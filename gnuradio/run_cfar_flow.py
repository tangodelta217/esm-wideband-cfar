#!/usr/bin/env python
"""Run a GNU Radio top block that applies CA-CFAR to FFT magnitude vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from gnuradio import blocks, fft, gr
    from gnuradio.fft import window
except ImportError:
    print("ERROR: GNU Radio no está instalado en este entorno.")
    print("Instalá GNU Radio (por ejemplo vía conda-forge) y volvé a ejecutar este script.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gr_blocks.cfar1d_block import ca_cfar_1d_block


class CFARFlowgraph(gr.top_block):
    """Top block que procesa un archivo IQ y aplica CA-CFAR sobre magnitud al cuadrado."""

    def __init__(
        self,
        iq_path: Path,
        sample_rate: float,
        center_freq: float,
        fft_size: int,
        num_vectors: int,
        num_train: int,
        num_guard: int,
        pfa: float,
    ) -> None:
        super().__init__("esm_wideband_cfar_flow")
        if not iq_path.exists():
            raise FileNotFoundError(iq_path)

        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.fft_size = fft_size
        self.num_vectors = num_vectors

        # Fuente IQ
        self.file_source = blocks.file_source(gr.sizeof_gr_complex, iq_path.as_posix(), repeat=False)

        # Convertir a vectores de FFT
        self.stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size)

        # Limitar cantidad de vectores si corresponde
        itemsize_vec = gr.sizeof_gr_complex * fft_size
        self.head = blocks.head(itemsize_vec, num_vectors) if num_vectors > 0 else None

        # Ventana y FFT
        win_taps = window.blackmanharris(fft_size)
        self.window_src = blocks.vector_source_c(win_taps, repeat=True, vlen=fft_size)
        self.window = blocks.multiply_vcc(fft_size)
        self.fft = fft.fft_vcc(fft_size, True, [], True)

        # Magnitud^2 y CFAR
        self.mag2 = blocks.complex_to_mag_squared(fft_size)
        self.cfar = ca_cfar_1d_block(
            vector_len=fft_size,
            num_train=num_train,
            num_guard=num_guard,
            pfa=pfa,
        )

        self.threshold_sink = blocks.vector_sink_f(fft_size)
        self.detection_sink = blocks.vector_sink_f(fft_size)
        self.power_sink = blocks.vector_sink_f(fft_size)

        # Conexiones
        self.connect(self.file_source, self.stream_to_vector)
        stream_node = self.stream_to_vector
        if self.head is not None:
            self.connect(self.stream_to_vector, self.head)
            stream_node = self.head

        self.connect(stream_node, (self.window, 0))
        self.connect(self.window_src, (self.window, 1))
        self.connect(self.window, self.fft)
        self.connect(self.fft, self.mag2)
        self.connect(self.mag2, self.power_sink)
        self.connect(self.mag2, self.cfar)

        self.connect((self.cfar, 0), self.threshold_sink)
        self.connect((self.cfar, 1), self.detection_sink)

    def log_results(self) -> None:
        detections = np.asarray(self.detection_sink.data(), dtype=np.float32)
        thresholds = np.asarray(self.threshold_sink.data(), dtype=np.float32)
        power = np.asarray(self.power_sink.data(), dtype=np.float32)
        if detections.size == 0:
            print("No se capturaron vectores (verificá parámetros de --vectors y archivo IQ).")
            return

        detections = detections.reshape((-1, self.fft_size))
        thresholds = thresholds.reshape((-1, self.fft_size))
        power = power.reshape((-1, self.fft_size))

        df = self.sample_rate / self.fft_size
        for vec_idx, det_vec in enumerate(detections):
            peak_bins = np.flatnonzero(det_vec > 0.5)
            if peak_bins.size == 0:
                continue
            print(f"Vector {vec_idx}:")
            for bin_idx in peak_bins:
                freq_hz = self.center_freq + (bin_idx - self.fft_size / 2) * df
                print(
                    f"  bin {bin_idx:4d} -> {freq_hz/1e6:.6f} MHz | "
                    f"power={power[vec_idx, bin_idx]:.3e} | "
                    f"threshold={thresholds[vec_idx, bin_idx]:.3e}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta un flujo GNU Radio con CA-CFAR.")
    parser.add_argument("--iq", type=Path, default=Path("data/iq_examples/tone_2msps_cf32.iq"), help="Ruta al archivo IQ cf32.")
    parser.add_argument("--fs", type=float, default=2_000_000.0, help="Frecuencia de muestreo (Hz).")
    parser.add_argument("--fc", type=float, default=915_000_000.0, help="Frecuencia central (Hz).")
    parser.add_argument("--fft", type=int, default=1024, help="Tamaño de la FFT / longitud de vector.")
    parser.add_argument("--vectors", type=int, default=200, help="Cantidad de vectores a procesar (0 = todo el archivo).")
    parser.add_argument("--train", type=int, default=16, help="Celdas de entrenamiento por lado.")
    parser.add_argument("--guard", type=int, default=4, help="Celdas de guarda por lado.")
    parser.add_argument("--pfa", type=float, default=1e-3, help="Probabilidad de falsa alarma objetivo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tb = CFARFlowgraph(
        iq_path=args.iq,
        sample_rate=args.fs,
        center_freq=args.fc,
        fft_size=args.fft,
        num_vectors=args.vectors,
        num_train=args.train,
        num_guard=args.guard,
        pfa=args.pfa,
    )
    tb.run()
    tb.log_results()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
