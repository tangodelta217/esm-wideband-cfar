from __future__ import annotations

import numpy as np
from gnuradio import gr

from src.cfar import ca_cfar_1d


class ca_cfar_1d_block(gr.sync_block):
    """GNU Radio block que aplica CA-CFAR sobre vectores de potencia."""

    def __init__(
        self,
        vector_len: int,
        num_train: int = 16,
        num_guard: int = 4,
        pfa: float = 1e-3,
    ) -> None:
        if vector_len <= 0:
            raise ValueError("vector_len debe ser positivo")
        gr.sync_block.__init__(
            self,
            name="ca_cfar_1d_block",
            in_sig=[(np.float32, vector_len)],
            out_sig=[(np.float32, vector_len), (np.float32, vector_len)],
        )
        self.vector_len = int(vector_len)
        self.num_train = int(num_train)
        self.num_guard = int(num_guard)
        self.pfa = float(pfa)

    def work(self, input_items, output_items):
        samples = input_items[0]
        out_thr = output_items[0]
        out_det = output_items[1]

        for idx, vec in enumerate(samples):
            power = np.asarray(vec, dtype=np.float64)
            det, thr = ca_cfar_1d(power, num_train=self.num_train, num_guard=self.num_guard, pfa=self.pfa)
            out_thr[idx][:] = thr.astype(np.float32, copy=False)
            out_det[idx][:] = det.astype(np.float32, copy=False)

        return len(samples)


__all__ = ["ca_cfar_1d_block"]
