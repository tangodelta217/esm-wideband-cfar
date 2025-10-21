# Importar primero GNU Radio (del sistema)
from gnuradio import gr, blocks, fft
from gnuradio.fft import window

# Recién ahora añadimos la raíz del repo para poder importar "src"
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.gr_blocks.cfar1d_block import cfar1d_block


def _cluster_detections(det_bins: np.ndarray, power_vec: np.ndarray, cluster_dist: int) -> np.ndarray:
    """Agrupa detecciones adyacentes y devuelve solo el pico de cada grupo."""
    if det_bins.size == 0:
        return np.array([], dtype=int)

    clusters = []
    current_cluster = [det_bins[0]]

    for i in range(1, det_bins.size):
        if det_bins[i] - det_bins[i-1] <= cluster_dist:
            current_cluster.append(det_bins[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [det_bins[i]]
    clusters.append(current_cluster)

    peak_bins = []
    for cluster in clusters:
        cluster_powers = power_vec[cluster]
        peak_idx_in_cluster = np.argmax(cluster_powers)
        peak_bins.append(cluster[peak_idx_in_cluster])

    return np.array(peak_bins, dtype=int)


class top_block(gr.top_block):
    def __init__(self, iq_path, fs=2e6, fc=100e6, nfft=4096, n_vectors=64):
        gr.top_block.__init__(self, "CFAR Flow")
        self.fs = fs; self.fc = fc; self.nfft = nfft; self.n_vectors = n_vectors
        self.num_train = 16
        self.num_guard = 4
        self.pfa = 1e-6  # Pfa ajustada para mayor robustez

        # Bloques
        self.src = blocks.file_source(gr.sizeof_gr_complex, iq_path, False)
        self.head = blocks.head(gr.sizeof_gr_complex, nfft * n_vectors)
        self.s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, nfft)
        self.fft = fft.fft_vcc(nfft, True, window.blackmanharris(nfft), True)
        self.c2m = blocks.complex_to_mag_squared(nfft)
        self.cfar = cfar1d_block(num_train=self.num_train, num_guard=self.num_guard, pfa=self.pfa, nfft=nfft)
        self.pwr_sink = blocks.vector_sink_f(nfft)
        self.thr_sink = blocks.vector_sink_f(nfft)
        self.det_sink = blocks.vector_sink_f(nfft)

        # --- Conexiones del Flowgraph ---
        # Cadena principal de procesamiento
        self.connect(self.src, self.head, self.s2v, self.fft, self.c2m, self.cfar)

        # Derivación (tap) para capturar la potencia para el clustering
        self.connect(self.c2m, self.pwr_sink)

        # Salidas del bloque CFAR
        self.connect((self.cfar, 0), self.thr_sink)
        self.connect((self.cfar, 1), self.det_sink)

    def run_and_report(self):
        """Ejecuta el flowgraph, aplica clustering y reporta los resultados."""
        start_time = time.perf_counter()
        self.run()
        end_time = time.perf_counter()

        # --- 1. Resultados de Detección ---
        power_vec = np.array(self.pwr_sink.data()[-self.nfft:], dtype=np.float32)
        det_vec = np.array(self.det_sink.data()[-self.nfft:], dtype=np.float32) > 0.5
        raw_bins = np.flatnonzero(det_vec)
        
        peak_bins = _cluster_detections(raw_bins, power_vec, cluster_dist=5)

        print("\n--- Resultados de Detección ---")
        print(f"Detecciones CFAR brutas: {len(raw_bins)}")
        print(f"Picos finales (post-clustering): {len(peak_bins)}")

        if peak_bins.size > 0:
            freqs = (peak_bins / self.nfft - 0.5) * self.fs + self.fc
            is_expected_tone = 2200 < peak_bins[0] < 2205
            print(f"Frecuencia(s) de picos (Hz): {np.round(freqs, 1)}")
            if is_expected_tone and len(peak_bins) == 1:
                print("Resultado: ✅ Coincide con la predicción de la prueba de humo.")
            else:
                print("Resultado: ❌ NO coincide con la predicción de la prueba de humo.")
        else:
            print("Detecciones: 0")
            print("Resultado: ❌ NO coincide con la predicción de la prueba de humo.")

        # --- 2. Métricas de Rendimiento ---
        print("\n--- Métricas de Rendimiento ---")
        total_time = end_time - start_time
        avg_latency_ms = (total_time / self.n_vectors) * 1000
        print(f"- Latencia media por trama: {avg_latency_ms:.3f} ms")

if __name__ == "__main__":
    iq_file = "data/iq_examples/tone_2msps_cf32.iq"
    tb = top_block(iq_file)
    tb.run_and_report()
