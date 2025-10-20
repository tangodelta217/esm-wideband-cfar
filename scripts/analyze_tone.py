import numpy as np

def analyze_iq(file_path, nfft=4096, fs=2e6):
    iq_data = np.fromfile(file_path, dtype=np.complex64)
    # Tomar una trama para el análisis
    frame = iq_data[:nfft]
    
    # Calcular la FFT con ventana y shift
    win = np.blackman(nfft)
    spectrum = np.fft.fftshift(np.fft.fft(frame * win)) / nfft
    power_db = 10 * np.log10(np.abs(spectrum)**2)
    
    # Encontrar el pico
    peak_bin = np.argmax(power_db)
    peak_freq = (peak_bin / nfft - 0.5) * fs
    peak_power_db = power_db[peak_bin]
    
    # Estimar el piso de ruido
    noise_bins = power_db[power_db < (peak_power_db - 10)] # Excluir el tono
    noise_floor_db = np.median(noise_bins)
    
    snr = peak_power_db - noise_floor_db
    
    print(f"Análisis de {file_path}:")
    print(f"- Frecuencia del Tono: {peak_freq / 1e3:.2f} kHz")
    print(f"- Bin del Tono (NFFT={nfft}): {peak_bin}")
    print(f"- SNR estimado: {snr:.1f} dB")

if __name__ == "__main__":
    analyze_iq("data/iq_examples/tone_2msps_cf32.iq")
