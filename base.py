import numpy as np
import scipy.signal as sc
import matplotlib.pyplot as plt
import NoiseFilter
import scipy.fft as ft
import scipy.optimize as opt
import scipy.stats as sts

# generating a sine wave + random noise for an image
noise_filter = NoiseFilter.NoiseFilter()
time, amplitude = noise_filter.sine_wave()
noisy_signal = noise_filter.random_noise(time, amplitude)

# applying Butterworth filter
order = 4
cutoff = 50
sampling_frequency = 1000
sos = sc.butter(order, cutoff, btype="lowpass", fs=sampling_frequency, output="sos")
filtered_signal = sc.sosfiltfilt(sos, noisy_signal)

'''
plt.plot(time, noisy_signal, label="Noisy Signal", alpha=0.5)
plt.plot(time, filtered_signal, label="Filtered Signal", linewidth=2)
plt.legend()
plt.show()
'''
# applying FFT to visualise frequency domain before/after
n = len(noisy_signal)
frequencies = ft.fftfreq(n, d=1/sampling_frequency)
half = n // 2
noise_fft = np.abs(ft.fft(noisy_signal))
filtered_fft = np.abs(ft.fft(filtered_signal))

plt.plot(frequencies[:half], noise_fft[:half], label="Noisy FFT", alpha=0.5)
plt.plot(frequencies[:half], filtered_fft[:half], label="Filtered FFT", linewidth=2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

# recovering original sine function
def sine_func(t, amp, fre, phase):
    return amp * np.sin(2 * np.pi * fre * t + phase)
params, _ = opt.curve_fit(sine_func, time, filtered_signal, p0=[1, 5, 0])

print("=== Signal Comparison ===")
print(f"{'':20} {'Noisy':>10} {'Filtered':>10}")
print(f"{'Mean':20} {np.mean(noisy_signal):>10.3f} {np.mean(filtered_signal):>10.3f}")
print(f"{'Std Dev':20} {np.std(noisy_signal):>10.3f} {np.std(filtered_signal):>10.3f}")
print(f"{'Max':20} {np.max(noisy_signal):>10.3f} {np.max(filtered_signal):>10.3f}")
print(f"{'Min':20} {np.min(noisy_signal):>10.3f} {np.min(filtered_signal):>10.3f}")

correlation_coefficient, _ = sts.pearsonr(amplitude, filtered_signal)
print(f"\nCorrelation with original: {correlation_coefficient:.4f}")