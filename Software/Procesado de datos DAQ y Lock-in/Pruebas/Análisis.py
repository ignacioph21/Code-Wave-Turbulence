from Formatter import read
from LockIn_fit_y_fft import lock_in
import numpy as np 
import matplotlib.pyplot as plt

ts, volts = read("../Test data/forzado.bin", plot_values=False) # RLC_attenuation.bin  
freq_Hz = 52e3
 
"""

N_lim = int(5000000 * 2.5)
A_t, phi_t = lock_in(ts[:, :N_lim], volts[:, :N_lim], cutoff_freq=1e1, plot_results=True) # , detrend=True 
A_t_r, phi_t_r = lock_in(np.array([ts[0], ts[0]])[:, :N_lim], np.array([volts[0], volts[0]])[:, :N_lim], cutoff_freq=1e1, plot_results=True) # , detrend=True 

plt.plot(ts[1, :N_lim], phi_t) # A
plt.plot(ts[0, :N_lim], phi_t_r)
plt.show()

plt.plot(ts[1, :N_lim], phi_t - phi_t_r) # Probar en FCD . Por ahí solo por ser tan baja la amplitud de la variación de la fase es tan importante el ruido de la propia modulación de la referencia. Lo raro es que aparezca solo después de los 20 segundos el mayor error en la referencia.
# O el common por ahí hacce cosas raras. Al darle a medir el osciloscopio mostraba cosas raras en sus mediciones de los mismos voltajes.
# Si uso SOLO la señal para los ajustes y sacar la referencia de ahí da lo mismo que el caso donde no resto después la referencia manualmente. Tal vez usar un modulador, random y aplicarlo a la referencia y al otro y restarlos, sin necesidad de tener que hacer un ajuste para la fase exacta, y la frecuencia de Fourier, o sea FFTs.
# Por ahí al hacer que haya más amplitud de fase cambiando la resistencia por una menor no es necesario fijarse en estas cosas. 

plt.show()
"""

"""
T = []
A = []
P = []

cutoff = int(1e3)
sample = int(5e5)
interval = 500000 # .0
for i in range(0, len(ts[0]), interval):
    print(i)
    t, v = ts[:, i:i+interval], volts[:, i:i+interval]
    A_t, phi_t = lock_in(t, v, cutoff_freq=cutoff) #    , detrend=True 

    T = np.append(T, t[0][20*cutoff:])
    A = np.append(A, A_t[20*cutoff:])
    P = np.append(P, phi_t[20*cutoff:])

plt.plot(T, P)
plt.show()
"""

# """ """
T = []
A = []
P = []

cutoff = int(1e3)
sample = int(5e5)
interval = 500000 
for i in range(interval//2, len(ts[0]), interval//2):
    print(i, i-interval//2, i+interval//2)
    t, v = ts[:, i-interval//2:i+interval//2], volts[:, i-interval//2:i+interval//2]
    A_t, phi_t = lock_in(t, v, cutoff_freq=cutoff) #    , detrend=True 

    T = np.append(T, t[0][interval//2:])
    A = np.append(A, A_t[interval//2:])
    P = np.append(P, phi_t[interval//2:])

# Si hago lockin para referencia contra referencia esperaríamos que de fase 0 constante, pero a partir de aproximadamente 20 segundos aparece una tendencia lineal, no sé porqué. Pasa en todos los juegos de mediciones. Probar aumentar el buffer. 

plt.plot(T, P) # TODO: Guardar en h5py para que no se vuelva más lento después de un par de iteraciones.
plt.vlines(range(interval//2, len(ts[0]), interval//2)*(t[0,1]-t[0,0]), -0.5, +0.5, linestyle="--", color="black", alpha=0.25)
plt.xlabel("Tiempo [s]")
plt.ylabel("Fase [rad]")
plt.show()
# """ """
