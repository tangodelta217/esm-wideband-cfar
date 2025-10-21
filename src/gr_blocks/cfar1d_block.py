from gnuradio import gr
import numpy as np
from src.cfar.cfar1d import ca_cfar_1d

class cfar1d_block(gr.sync_block):
    """
    Recibe vector de potencia (float32, vlen=nfft) y emite:
    - salida 0: threshold (float32, vlen=nfft)
    - salida 1: detections (float32 0/1, vlen=nfft)
    """
    def __init__(self, num_train=16, num_guard=4, pfa=1e-3, nfft=4096):
        gr.sync_block.__init__(
            self,
            name="cfar1d_block",
            in_sig=[(np.float32, nfft)],
            out_sig=[(np.float32, nfft), (np.float32, nfft)],
        )
        self.num_train = int(num_train)
        self.num_guard = int(num_guard)
        self.pfa = float(pfa)
        self.nfft = int(nfft)

    def work(self, input_items, output_items):
        # input_items[0] -> shape (1, nfft)
        P = input_items[0][0].astype(np.float64, copy=False)
        det, thr = ca_cfar_1d(P, self.num_train, self.num_guard, self.pfa)
        output_items[0][0][:] = thr.astype(np.float32, copy=False)
        output_items[1][0][:] = det.astype(np.float32, copy=False)
        return 1

__all__ = ["cfar1d_block"]
