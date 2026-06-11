import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

class NoiseFilter:
    def __init__(self):
        self._frequency = 5
        self._duration = 1
        self._sample_rate = 1000

    def sine_wave(self):
        time = np.linspace(0, self._duration, int(self._sample_rate * self._duration), endpoint=False)
        amplitude = np.sin(2 * np.pi * self._frequency * time)

        return time, amplitude
    
    def random_noise(self, time, amplitude):
        rng = np.random.default_rng()
        ran_noise = rng.standard_normal(len(time))
        noisy_signal = amplitude + ran_noise

        return noisy_signal