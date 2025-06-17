from Formatter import read
from LockIn_fit_y_fft import lock_in
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import detrend

ts, volts = read("Test data/forzado.bin", plot_values=False) # RLC_attenuation.bin  

alpha = 0.01
t = [ts[0, 0]]
for i in range(len(ts[0, 1:])):
    if i%500000 == 0:
        print(i)
    t.append(t[i-1] + alpha * (ts[0, i] - t[i-1]) )

