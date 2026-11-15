"""Audio loading shared by the native service, using its existing dependencies."""
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_mono_audio(path, sample_rate=16000):
    samples, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Reference audio must contain finite samples')
    samples = samples.mean(axis=1)
    if rate != sample_rate:
        divisor = gcd(rate, sample_rate)
        samples = resample_poly(samples, sample_rate // divisor, rate // divisor)
    return np.asarray(samples, dtype='<f4')
