from Formatter import read
from LockIn_fit_y_fft import lock_in
import numpy as np 
import matplotlib.pyplot as plt

ts, volts = read("../Test data/forzado.bin", plot_values=False) # RLC_attenuation.bin  
freq_Hz = 52e3
 


N_lim = int(5000000 * 2.5)
A_t, phi_t = lock_in(ts[:, :N_lim], volts[:, :N_lim], cutoff_freq=1e1) # , detrend=True , plot_results=True
A_t_r, phi_t_r = lock_in(np.array([ts[0], ts[0]])[:, :N_lim], np.array([volts[0], volts[0]])[:, :N_lim], cutoff_freq=1e1) # , detrend=True , plot_results=True      



n = 1000
fig, axs = plt.subplots(2, 1, sharey=True, sharex=True)

axs[0].plot(ts[1, :N_lim][::n], phi_t[::n], label="Fase Señal") 
axs[0].plot(ts[0, :N_lim][::n], phi_t_r[::n], label="Fase referencia")
axs[0].legend()
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Fase [rad]")

axs[1].set_title("Diferencia $\phi_r-\phi_{ref}$.")
axs[1].plot(ts[1, :N_lim][::n], (phi_t - phi_t_r)[::n]) # Probar en FCD . Por ahí solo por ser tan baja la amplitud de la variación de la fase es tan importante el ruido de la propia modulación de la referencia. Lo raro es que aparezca solo después de los 20 segundos el mayor error en la referencia.
# O el common por ahí hacce cosas raras. Al darle a medir el osciloscopio mostraba cosas raras en sus mediciones de los mismos voltajes.
# Si uso SOLO la señal para los ajustes y sacar la referencia de ahí da lo mismo que el caso donde no resto después la referencia manualmente. Tal vez usar un modulador, random y aplicarlo a la referencia y al otro y restarlos, sin necesidad de tener que hacer un ajuste para la fase exacta, y la frecuencia de Fourier, o sea FFTs.
# Por ahí al hacer que haya más amplitud de fase cambiando la resistencia por una menor no es necesario fijarse en estas cosas. 
axs[1].set_xlabel("Tiempo [s]")
axs[1].set_ylabel("Fase [rad]")

plt.tight_layout()
plt.show()
