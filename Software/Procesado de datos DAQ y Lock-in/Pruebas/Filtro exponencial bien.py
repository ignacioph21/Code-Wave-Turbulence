from Formatter import read
from LockIn_fit_y_fft import lock_in
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import detrend
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import os
import re
from Formatter import read
from scipy.signal import detrend as linear_detrend

def reconstruct_reference(ts_reference, reference):
    def sin(t, A, phi, f):
        return A * np.sin(2 * np.pi * f * t + phi)

    freqs = np.fft.rfftfreq(len(ts_reference), d=(ts_reference[1]-ts_reference[0]))
    Y = np.fft.rfft(reference)    
    idx = np.argwhere(np.abs(Y) == np.max(np.abs(Y)))[0][0]
    f_est = freqs[idx]

    p0 = [np.max(reference), 1.50, f_est]
    popt, povc = curve_fit(sin, ts_reference, reference, p0=p0)

    residuals = reference - sin(ts_reference, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((reference-np.mean(reference))**2)
    r_squared = 1 - (ss_res / ss_tot)

    if r_squared < 0.990:
        print(f"The R^2 for the reference signal with f = {f_est:.1f} is low ({r_squared:.5f}).")

    # plt.show()

    return popt

ts, volts = read("Test data/forzado.bin", plot_values=False) # RLC_attenuation.bin  

ts_reference = ts[0]
ts_signal = ts[1]

reference = volts[0]
signal = volts[1]

A_ref, phi_ref, f_ref = reconstruct_reference(ts_reference, reference) # TODO: f_Est de FFT.

ref_cos = np.sign(A_ref) * np.cos(2*np.pi*f_ref * ts_signal + phi_ref) # IMPORTANTE SIGNO. - 
ref_sin = np.sign(A_ref) * np.sin(2*np.pi*f_ref * ts_signal + phi_ref)


X0 = ref_sin*signal
Y0 = ref_cos*signal # S 

fd = 5e5
fc = 1e3

gamma = 2*np.pi*fc/fd
alpha = np.cos(gamma) - 1 + np.sqrt(np.cos(gamma)**2 - 4*np.cos(gamma) + 3)

X1 = [0]
Y1 = [0]
X2 = [0]
Y2 = [0]

for i in range(len(ts[0])):
    if i%500000 == 0:
        print(i)
    X1.append(X1[i-1] + alpha * (X0[i] - X1[i-1]) )
    Y1.append(Y1[i-1] + alpha * (Y0[i] - Y1[i-1]) )
    X2.append(X2[i-1] + alpha * (X1[i] - X2[i-1]) )
    Y2.append(Y2[i-1] + alpha * (Y1[i] - Y2[i-1]) )

plt.plot(ts[0], np.arctan2(X2, Y2))
plt.show()
