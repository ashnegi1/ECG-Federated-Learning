"""Configurable zero-phase Butterworth and wavelet ECG preprocessing."""
from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt


def butterworth_bandpass(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError(f"Require 0 < low_hz < high_hz < Nyquist ({fs / 2:g} Hz)")
    sos = butter(order, (low_hz, high_hz), btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=np.float64)).astype(np.float32)


def dwt_denoise(signal: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    wave = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(x), wave.dec_len)
    use_level = min(level, max_level)
    if use_level < 1:
        return x.astype(np.float32, copy=True)
    coeffs = pywt.wavedec(x, wave, mode="symmetric", level=use_level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6744897501960817
    threshold = sigma * np.sqrt(2.0 * np.log(max(len(x), 2)))
    denoised = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    reconstructed = pywt.waverec(denoised, wave, mode="symmetric")[: len(x)]
    return reconstructed.astype(np.float32)


def preprocess_ecg(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 45.0,
    filter_order: int = 4,
    wavelet: str = "db4",
    wavelet_level: int = 4,
    **_: object,
) -> np.ndarray:
    filtered = butterworth_bandpass(signal, fs, low_hz, high_hz, filter_order)
    return dwt_denoise(filtered, wavelet, wavelet_level)
