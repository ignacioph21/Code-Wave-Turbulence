import numpy as np
import matplotlib.pyplot as plt
import os

# Parámetros físicos
g = 9.81          # gravedad (m/s²)
d = 3.50 / 100     # profundidad en m  4.7 
sigma = 0.0740     # tensión superficial N/m
rho = 997         # densidad del agua kg/m³

w = (20.0) / 100 
h = (20.0) / 100 

# Función para calcular la frecuencia asociada a una longitud de onda l
def freq(k):  
    omega = np.sqrt((g * k + (sigma / rho) * k**3) * np.tanh(k * d))
    return omega / (2 * np.pi)

# Calcular frecuencias teóricas para los primeros 10 múltiplos de lambda_base
ns = [0, 1, 2, 3, 4, 5]   
ms = [0, 1]
freqs_pair = [] 

frecuencias_teoricas = []
for n in ns:
    for m in ms:
        kx = np.pi * n / w
        ky = np.pi * m / h
        k = np.sqrt(kx**2 + ky**2)
    
        f_mn = freq(k)
        frecuencias_teoricas.append(f_mn)
        freqs_pair.append(f"(n = {n}, m = {m})") # a 

# Cargar datos
data = np.load("rotar_caja2.npy") # $"data_forzado.npy"

# FFT
start = 2000
end = -1
times = data[0][start:end]
datos = data[1][start:end]

N = len(times) 
fs = 1/(times[1]-times[0])

fft_vals = np.fft.fft(datos - np.mean(datos))  # quitar media
fft_freqs = np.fft.fftfreq(N, d=1/fs)

# Tomar mitad positiva
fft_vals = fft_vals[:N//2]
fft_freqs = fft_freqs[:N//2]
fft_magnitude = np.abs(fft_vals)

plt.subplot(211)
plt.plot(times, datos, ".-")
plt.xlabel("Tiempo [s]")
plt.ylabel("Fase $\phi$ [°]")


plt.subplot(212)
# --- Graficar FFT ---
plt.plot(fft_freqs, fft_magnitude, label='FFT de señal', color='blue')

# Añadir líneas verticales para las frecuencias teóricas
for i, f in enumerate(frecuencias_teoricas): 
    plt.axvline(x=f, color='red', linestyle='--', alpha=0.6)
    plt.text(f, max(fft_magnitude)*0.9, freqs_pair[i], rotation=90, verticalalignment='top', horizontalalignment='right', fontsize=8-6+2-1+2, color='red') # # 0.99* 
    plt.xlim(-0.1, 50-40+10)          

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.legend()
# plt.tight_layout()
plt.show()
