import matplotlib.pyplot as plt

from mag_utils.functional.signal_processing import lowpass_filter, bandpass_filter, highpass_filter
import numpy as np

SAMPLE_RATE = 10_000


def mock_signals(sample_rate=SAMPLE_RATE):
    t = np.arange(0, 1000, 1 / sample_rate)

    sig1 = 5 * np.sin(2 * 2 * np.pi * t)
    sig2 = 5 * np.sin(20 * 2 * np.pi * t)

    return sig1, sig2


def test_lp_filter_wrapper():
    sig1, sig2 = mock_signals()
    filtered_signal = lowpass_filter(sig1 + sig2, 5, 15, SAMPLE_RATE)

    np.testing.assert_array_almost_equal(sig1, filtered_signal, decimal=1)


def test_hp_filter_wrapper():
    sig1, sig2 = mock_signals()
    filtered_signal = highpass_filter(sig1 + sig2, 5, 15, SAMPLE_RATE)

    np.testing.assert_array_almost_equal(filtered_signal, sig2, decimal=1)


def test_bp_filter_wrapper():
    sig1, sig2 = mock_signals()
    filtered_signal = bandpass_filter(sig1 + sig2, 15, 25, SAMPLE_RATE)

    assert sum(filtered_signal / sig2 > 2) / len(filtered_signal) < 1e-2, "Mismatch between signals is too large"


