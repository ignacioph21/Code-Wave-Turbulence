from Formatter import read
from LOCK_IN import lock_in
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import detrend
import time

ts, volts = read("../Test data/forzado.bin", plot_values=False) # RLC_attenuation.bin  

T = []
P = []

cutoff = int(1e3)
sample = 5e5 
interval = 50000
for i in range(0, len(ts[0]), interval):
    t0 = time.time()

    if i % 500000 == 0:
        print(i)
    t, v = ts[:, i:i+interval], volts[:, i:i+interval]
    A_t, phi_t = lock_in(t, v, cut_off=cutoff)  

    T = np.append(T, t[0][-1])
    P = np.append(P, phi_t[-1])

    t1 = time.time()
    # print(f"Elapsed {t1-t0} s to calculate point of {interval/sample} s.")

plt.plot(T, P, ".-") # TODO: Guardar en h5py para que no se vuelva más lento después de un par de iteraciones.
plt.show() # Le da justo el tiempo para calcular en tiempo real este intervalo.
