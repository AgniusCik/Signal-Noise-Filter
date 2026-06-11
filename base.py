import numpy as np
import scipy.signal as sc
import matplotlib.pyplot as plt
import NoiseFilter
import scipy.fft as fft

# generating a sine wave + random noise for an image
noise_filter = NoiseFilter.NoiseFilter()
time, amplitude = noise_filter.sine_wave()
noisy_signal = noise_filter.random_noise(time, amplitude)

# applying filters to regain the original sine wave
order = 4
cutoff = 50
sampling_frequency = 1000
sos = sc.butter(order, cutoff, btype="lowpass", fs=sampling_frequency, output="sos")
filtered_signal = sc.sosfiltfilt(sos, noisy_signal)

plt.plot(time, noisy_signal, label="Noisy Signal", alpha=0.5)
plt.plot(time, filtered_signal, label="Filtered Signal", linewidth=2)
plt.legend()
plt.show()

# applying FFT to visualise frequency domain before/after
