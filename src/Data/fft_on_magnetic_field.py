from mag_utils.mag_utils import load
import numpy as np
import matplotlib.pyplot as plt


def fourier_transform(x, y, B, out):

    # Apply FFT to data
    fft = np.fft.fft(B)

    # Identify frequencies with high amplitudes
    amplitudes = np.abs(fft)
    threshold = 4 * np.std(amplitudes)
    anomalies = np.where(amplitudes > threshold)[0]

    # Create a copy of the FFT and remove all other frequency components
    fft_only_anomalies = np.zeros_like(fft)
    fft_only_anomalies[anomalies] = fft[anomalies]

    anomalies_data = np.fft.ifft(fft_only_anomalies).real

    plt.figure(0)
    # Plot the original data, the identified anomalies, and the anomalies-only data
    plt.plot(range(len(B)), B, label='Original Data')
    plt.plot(anomalies, B[anomalies], 'ro', label='Anomalies')
    plt.plot(range(len(B)), np.zeros_like(B) + np.nan, 'k--')  # add a horizontal line for clarity
    plt.plot(range(len(B)), anomalies_data, 'g', label='Anomalies-only Data_IFFT')
    plt.legend()
    plt.savefig(f"{out}/find_anomaly_by_fft.png")










