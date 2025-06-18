import numpy as np 
import matplotlib.pyplot as plt
from pyroomacoustics.utilities import fractional_delay
from scipy.signal import convolve
from scipy.signal import butter, filtfilt
from Formatter import read

def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def fft_parameters(ts, signal):
    fs = 1/(ts[1] - ts[1-1])
    freqs = np.fft.rfftfreq(len(ts), d=1/fs)
    Y = 2 * np.fft.rfft(signal) / len(ts)   
    idx = np.argwhere(np.abs(Y) == np.max(np.abs(Y)))[0][0]
    fr = freqs[idx]
    Ar = np.abs(Y[idx])
    return Ar, fr


def resample(ts, signal):
    fs = 1/(ts[1] - ts[1-1])
    t0 = -0.5
    h = fractional_delay(t0)
    delayed = convolve(signal, h, mode="same")
    new_ts = ts - t0/fs
    return new_ts, delayed

def shift_90(ts, signal):
    fs = 1/(ts[1] - ts[1-1])

    Ar, fr = fft_parameters(ts, signal)
        
    Delta = -1/4 * fs/fr # 90° shift.
    h_cos = fractional_delay(Delta)
    cos_measured = convolve(signal, h_cos, mode="same")
    return cos_measured

def get_amplitude_and_phase(ts_reference, reference, ts_signal, signal, cut_off=10, check_progress=False):
    _, sin_ref = resample(ts_reference, reference)
    _, cos_ref = resample(ts_reference, shift_90(ts_reference, reference))

    A, _ = fft_parameters(ts_reference, reference)
    if (Delta := abs(A - np.max(reference))/abs(A)) > 0.1:
        print(f"Maybe amplitud normalization with fft failed, Relative Difference is {Delta*100:.2f} %.")
    
    X0 = 2 * cos_ref  / A * signal # Debe haber mejor forma de determinar la amplitud. FFT tal vez.
    Y0 = 2 * sin_ref  / A * signal

    fs = 1 / (ts_reference[1] - ts_reference[0])
    b, a = butter_lowpass(cut_off, fs)
    X1 = filtfilt(b, a, X0)
    Y1 = filtfilt(b, a, Y0)

    return np.sqrt(np.array(X1)**2 + np.array(Y1)**2), np.arctan2(X1, Y1)       

def lock_in(times, volts, cut_off=10, plot_results=False, check_progress=False):
    t_ref, volts_ref = times[0], volts[0]  # referencia: canal 1
    t_signal, volts_signal = times[1], volts[1]  # señal: canal 2

    A, phi = get_amplitude_and_phase(t_ref, volts_ref, t_signal, volts_signal, cut_off, check_progress)

    if plot_results:
        plt.subplot(211)
        plt.plot(t_signal, A) 
        plt.ylabel("Amplitud [V]")
        plt.subplot(212)
        plt.plot(t_signal, phi)  # _t 
        plt.ylabel("Fase [rad]")
        plt.xlabel("Tiempo [s]")
        plt.show()

    return A, phi


if __name__ == "__main__":
    t, v = read("nuevo_forzado360Ohm.bin", check_metadata=True) # nuevo_frozado_10Ohm.bin 
    A, phi = lock_in(t, v, cut_off=1000)
    plt.plot(t[0], phi)
    plt.show()
