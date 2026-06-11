import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import NoiseFilter

# generating a sine wave + random noise for an image
noise_filter = NoiseFilter.NoiseFilter()
time, amplitude = noise_filter.sine_wave()
noisy_signal = noise_filter.random_noise(time, amplitude)

fig, ax = plt.subplots()
ax.plot(time, noisy_signal)
plt.show()