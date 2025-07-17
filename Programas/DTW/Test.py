import numpy as np
import pywt
import matplotlib.pyplot as plt
import random
import yfinance as yf


def FFT_plot(signal, sample_rate=1000):
    sample_rate = sample_rate
    N = len(signal)

    # Remove DC offset (mean subtraction)
    signal_centered = signal - np.mean(signal)
    fft_result = np.fft.fft(signal_centered)
    frequencies = np.fft.fftfreq(N, 1 / sample_rate)
    magnitude = np.abs(fft_result) * 2 / N  # Scaled magnitude

    plt.figure()
    plt.plot(frequencies[: N // 2], magnitude[: N // 2])
    plt.title(f"FFT for signal {signal.name}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.show()


def FFT(signal, sample_rate=1000):
    N = len(signal)
    signal_centered = signal - np.mean(signal)  # Remove DC offset
    fft_result = np.fft.fft(signal_centered)  # Full complex FFT
    frequencies = np.fft.fftfreq(N, 1 / sample_rate)
    full_fft = np.fft.fft(signal_centered)

    # Extract positive frequencies (one-sided FFT)
    if N % 2 == 0:
        pos_freq = frequencies[: N // 2]  # Frequencies (0 to Nyquist)
        pos_fft = full_fft[: N // 2]  # Complex FFT values (one-sided)
    else:
        pos_freq = frequencies[: (N // 2 + 1)]  # For odd N
        pos_fft = full_fft[: (N // 2 + 1)]

    # Magnitude for plotting (optional)
    magnitude = np.abs(pos_fft)  # Scaled magnitude

    return pos_freq, magnitude, pos_fft, full_fft


# Extraccion de datos de yfinance

tickers = ["AAPL"]
start_date = "2024-01-01"
end_date = "2024-04-02"

data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
data = data.dropna()

FFT_plot(data["AAPL"])
plt.figure(figsize=(10, 5))
plt.plot(data["AAPL"])
plt.show()

coeffs = pywt.wavedec(data["AAPL"], "db4", level=10)

level = 10
fig, axs = plt.subplots(level + 2, 1, figsize=(16, 2 * (level + 2)))

# Plot original signal
axs[0].plot(data["AAPL"])
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Amplitude")

# Plot approximation coefficients
axs[1].plot(coeffs[0])
axs[1].set_title("Approximation Coefficients (Level 10)")
axs[1].set_xlabel("Sample")
axs[1].set_ylabel("Amplitude")

# Plot detail coefficients for each level
for i in range(level):
    axs[i + 2].plot(coeffs[i + 1])
    axs[i + 2].set_title(f"Detail Coefficients (Level {level-i})")
    axs[i + 2].set_xlabel("Sample")
    axs[i + 2].set_ylabel("Amplitude")

plt.tight_layout()
plt.show()
