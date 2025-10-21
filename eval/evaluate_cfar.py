from eval.pd_tools import sweep_snr

if __name__ == "__main__":
    res = sweep_snr(range(-10, 21, 2))
    for snr, pd, pfa_emp in res:
        print(f"SNR={snr:>3} dB  Pd={pd:0.3f}  (Pfa_emp ~ {pfa_emp:0.2e})")
