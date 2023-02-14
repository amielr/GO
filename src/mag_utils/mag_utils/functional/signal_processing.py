import numpy as np
import scipy.signal as sp
import sk_dsp_comm.fir_design_helper as fir

# set default sample rate
SAMPLE_RATE_HZ = 10

def calculate_fft(signal: np.array, sample_rate=SAMPLE_RATE_HZ):
    """
    Args:
        signal: the signal to be analyzed in the frequency domain.
        sample_rate: the sample rate of the data (in Hz).

    Returns: ftt frequencies (x-axis), fft amplitude (y-axis)

    """
    fft_freq = np.fft.fftfreq(len(signal), d=1 / sample_rate)
    fft_signal = np.abs(np.fft.fft(signal - signal.mean()))

    return fft_freq, fft_signal


def lowpass_filter(signal: np.array, f_pass=0.5, f_stop=1, sample_rate=SAMPLE_RATE_HZ):
    """
    Args:
        signal: the signal to be filtered
        f_pass: the frequency from which the filtered signal will be diminished
        f_stop: the frequency until which the filtered signal will be diminished
        From that freq on, all frequencies are excluded.
        sample_rate: the sample rate of the data (in Hz).

    Note: since this is a lowpass filter, f_pass must be less than f_stop.

    Returns: filtered signal in a np.array

    """
    assert f_pass < f_stop, "f_pass must be less than f_stop."
    b_filter = fir.fir_remez_lpf(f_pass=f_pass, f_stop=f_stop, d_pass=0.1, d_stop=40, fs=sample_rate)

    return sp.filtfilt(b=b_filter, a=1, x=signal)


def highpass_filter(signal: np.array, f_stop=2, f_pass=1, sample_rate=SAMPLE_RATE_HZ) -> np.array:
    """
       Args:
           signal: the signal to be filtered
           f_pass: the frequency from which the filtered signal will be diminished
           f_stop: the frequency until which the filtered signal will be diminished.
           From that freq on, all frequencies are excluded.
           sample_rate: the sample rate of the data (in Hz).

       Note: since this is a highpass filter, f_pass must be greater than f_stop.

       Returns: filtered signal in a np.array
    """
    assert f_stop < f_pass, "f_stop must be less than f_pass."
    b_filter = fir.fir_remez_hpf(f_pass=f_pass, f_stop=f_stop, d_pass=0.1, d_stop=40, fs=sample_rate)

    return sp.filtfilt(b=b_filter, a=1, x=signal)


def bandpass_filter(signal, f_pass=0.5, f_stop=1, sample_rate=SAMPLE_RATE_HZ) -> np.array:
    """
           Args:
               signal: the signal to be filtered.
               f_pass: the frequency from which all frequencies will pass.
               f_stop: the frequency until which all frequencies will pass.
               From that freq on, all frequencies are excluded.
               sample_rate: the sample rate of the data (in Hz).

           Note: since this is a bandpass filter, f_pass must be less than f_stop.

           Returns: filtered signal in a np.array
    """
    assert f_pass < f_stop, "f_pass must be less than f_stop."
    nyq_freq = sample_rate / 2
    f_pass, f_stop = f_pass / nyq_freq, f_stop / nyq_freq
    b_filter = sp.butter(5, [f_pass, f_stop], btype="bandpass", output="sos")

    return sp.sosfiltfilt(b_filter, signal, axis=0)
