from Formatter import read
from LockIn_fit_y_fft import lock_in
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import detrend

ts, volts = read("Test data/forzado.bin", plot_values=False) # RLC_attenuation.bin  

T = []
P = []

cutoff = int(1e3)
sample = int(5e5)
interval = 50000 
for i in range(0, len(ts[0]), interval):
    if i % 500000 == 0:
        print(i)
    t, v = ts[:, i:i+interval], volts[:, i:i+interval]
    try:
        A_t, phi_t = lock_in(t, v, cutoff_freq=cutoff)  

        T = np.append(T, t[0][interval//2])
        P = np.append(P, detrend(phi_t[-interval//2:])[-1])
    except:
        print("Weird fit.")
        pass

# plt.plot(t[0+1], phi_t)
# plt.show()
# Si hago lockin para referencia contra referencia esperaríamos que de fase 0 constante, pero a partir de aproximadamente 20 segundos aparece una tendencia lineal, no sé porqué. Pasa en todos los juegos de mediciones. Probar aumentar el buffer. 

plt.plot(T, P, ".-") # TODO: Guardar en h5py para que no se vuelva más lento después de un par de iteraciones.
plt.show()
