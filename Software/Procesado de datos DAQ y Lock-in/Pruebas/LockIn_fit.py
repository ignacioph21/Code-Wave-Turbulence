import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import os
import re
from Formatter import read
from scipy.signal import detrend as linear_detrend

def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def reconstruct_reference(ts_reference, reference, f_est):
    def sin(t, A, phi, f):
        return A * np.sin(2 * np.pi * f * t + phi)

    p0 = [np.max(reference), 1.50, f_est]
    popt, povc = curve_fit(sin, ts_reference, reference, p0=p0)

    residuals = reference - sin(ts_reference, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((reference-np.mean(reference))**2)
    r_squared = 1 - (ss_res / ss_tot)

    if r_squared < 0.990:
        print(f"The R^2 for the reference signal with f = {f_est:.1f} is low ({r_squared:.5f}).")

    plt.show()

    return popt


def get_amplitude_and_phase(ts_reference, reference, ts_signal, signal, f_est, cutoff_freq=10):
    fs = 1 / (ts_signal[1] - ts_signal[0])

    A_ref, phi_ref, f_ref = reconstruct_reference(ts_reference, reference, f_est) # TODO: f_Est de FFT.

    # Construir referencias en los tiempos de la señal
    ref_cos = np.sign(A_ref) * np.cos(2*np.pi*f_ref * ts_signal + phi_ref) # IMPORTANTE SIGNO. - 
    ref_sin = np.sign(A_ref) * np.sin(2*np.pi*f_ref * ts_signal + phi_ref)
    
    in_phase = signal * ref_sin
    quadrature = signal * ref_cos

    b, a = butter_lowpass(cutoff_freq, fs)
    in_phase_filtered = filtfilt(b, a, in_phase)
    quadrature_filtered = filtfilt(b, a, quadrature)

    A_t = 2 * np.sqrt(in_phase_filtered**2 + quadrature_filtered**2)
    phi_t = np.arctan2(quadrature_filtered, in_phase_filtered)

    return A_t, phi_t

def lock_in(times, volts, f_est, cutoff_freq=10, detrend=False, plot_results=False): # Para DAQ con canal 1 referencia y canal 2 señal.
    t_ref, volts_ref = times[0], volts[0]  # referencia: canal 1
    t_signal, volts_signal = times[1], volts[1]  # señal: canal 2

    A_t, phi_t = get_amplitude_and_phase(t_ref, volts_ref, t_signal, volts_signal, f_est, cutoff_freq=1e2)
    phi_t = linear_detrend(np.unwrap(phi_t)) if detrend else np.unwrap(phi_t)

    if plot_results:
        plt.subplot(211)
        plt.plot(t_signal, A_t) # , ".-" 
        plt.ylabel("Amplitud [V]")
        plt.subplot(212)
        plt.plot(t_signal, phi_t) # , ".-" 
        plt.ylabel("Fase [rad]")
        plt.xlabel("Tiempo [s]")
        plt.show()

    return A_t, phi_t


