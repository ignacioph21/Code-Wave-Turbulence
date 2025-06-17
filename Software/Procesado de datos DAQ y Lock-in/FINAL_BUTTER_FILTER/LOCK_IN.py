from Formatter import read
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

def resample(ts, signal):
    fs = 1/(ts[1] - ts[1-1])
    t0 = -0.5
    h = fractional_delay(t0)
    delayed = convolve(signal, h, mode="same")
    new_ts = ts - t0/fs
    return new_ts, delayed

def shift_90(ts, signal):
    fs = 1/(ts[1] - ts[1-1])

    freqs = np.fft.rfftfreq(len(ts), d=1/fs)
    Y = np.fft.rfft(signal)    
    idx = np.argwhere(np.abs(Y) == np.max(np.abs(Y)))[0][0]
    fr = freqs[idx]
    
    Delta = -1/4 * fs/fr # 90° shift.
    h_cos = fractional_delay(Delta)
    cos_measured = convolve(signal, h_cos, mode="same")
    return cos_measured

def get_amplitude_and_phase(ts_reference, reference, ts_signal, signal, cut_off=10, check_progress=False):
    _, sin_ref = resample(ts_reference, reference)
    _, cos_ref = resample(ts_reference, shift_90(ts_reference, reference))

    X0 = 2 * cos_ref  / np.max(reference) * signal # Debe haber mejor forma de determinar la amplitud. FFT tal vez.
    Y0 = 2 * sin_ref  / np.max(reference) * signal

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
        plt.plot(t_signal, phi_t)  
        plt.ylabel("Fase [rad]")
        plt.xlabel("Tiempo [s]")
        plt.show()

    return A, phi

if __name__ == "__main__":
    ts, volts = read("../Test data/forzado.bin", plot_values=False) #   RLC_attenuation.bin 

    # TODO: guardar en archivo h5py los X1, Y1, X2, Y2 (solo los últimos en realidad) y que queden en memoria los valores anteriores solo para acelerar.
    A, phi = lock_in(ts, volts, 1e3, check_progress=True)
    plt.plot(ts[0], phi)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Fase [rad]")
    # plt.legend()
    plt.show()
