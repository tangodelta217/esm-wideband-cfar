# Importar primero GNU Radio (del sistema)
from gnuradio import gr, blocks, fft
from gnuradio.fft import window

# Recién ahora añadimos la raíz del repo para poder importar "src"
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.gr_blocks.cfar1d_block import cfar1d_block

class top_block(gr.top_block):
    def __init__(self, iq_path, fs=2e6, fc=100e6, nfft=4096, n_vectors=64):
        gr.top_block.__init__(self, "CFAR Flow")
        self.fs = fs; self.fc = fc; self.nfft = nfft

        # Fuente IQ (cf32), flujo finito
        self.src  = blocks.file_source(gr.sizeof_gr_complex, iq_path, False)
        self.head = blocks.head(gr.sizeof_gr_complex, nfft * n_vectors)

        # STFT: vector -> FFT (ventana Blackman-Harris, con shift interno)
        self.s2v  = blocks.stream_to_vector(gr.sizeof_gr_complex, nfft)
        self.fft  = fft.fft_vcc(nfft, True, window.blackmanharris(nfft), True)

        # |·|^2 por bin
        self.c2m  = blocks.complex_to_mag_squared(nfft)

        # CFAR 1D sobre vector de potencias
        self.cfar = cfar1d_block(num_train=16, num_guard=4, pfa=1e-3, nfft=nfft)

        # Sinks para leer el último vector
        self.thr_sink = blocks.vector_sink_f(nfft)
        self.det_sink = blocks.vector_sink_f(nfft)

        # Conexiones
        self.connect(self.src, self.head, self.s2v, self.fft, self.c2m, self.cfar)
        self.connect((self.cfar, 0), self.thr_sink)
        self.connect((self.cfar, 1), self.det_sink)

    def run_once(self):
        self.run()
        thr = np.array(self.thr_sink.data()[-self.nfft:], dtype=np.float32)
        det = np.array(self.det_sink.data()[-self.nfft:], dtype=np.float32) > 0.5
        bins = np.flatnonzero(det)
        if bins.size:
            freqs = (bins / self.nfft - 0.5) * self.fs + self.fc
            print(f"Detections: {len(freqs)} | First Hz: {np.round(freqs[:5], 1)}")
        else:
            print("Detections: 0")

if __name__ == "__main__":
    tb = top_block("data/iq_examples/tone_2msps_cf32.iq", fs=2e6, fc=100e6, nfft=4096, n_vectors=64)
    tb.run_once()
