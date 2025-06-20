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

def resample(ts, signal, t0=-0.5):
    fs = 1/(ts[1] - ts[1-1])
    # t0 = -0.5
    h = fractional_delay(t0)
    delayed = convolve(signal, h, mode="same")
    new_ts = ts - t0/fs
    return new_ts, delayed

def shift_90(ts, signal, fr):
    fs = 1/(ts[1] - ts[1-1])
        
    Delta = -1/4 * fs/fr # 90° shift.
    h_cos = fractional_delay(Delta)
    cos_measured = convolve(signal, h_cos, mode="same")
    return cos_measured

def low_pass_filter(X0, Y0, cut_off,  fs, mode="butter", check_progress=False):
    if mode == "butter":
        b, a = butter_lowpass(cut_off, fs)
        X = filtfilt(b, a, X0)
        Y = filtfilt(b, a, Y0)

    if mode == "exponential":
        gamma = 2 * np.pi * cut_off / fs
        alpha = np.cos(gamma) - 1 + np.sqrt(np.cos(gamma)**2 - 4*np.cos(gamma) + 3)

        X1, Y1 = 0, 0
        X, Y = [0], [0]

        for i in range(len(X0)):
            if check_progress:
                if i % (len(X0) // 100) == 0:
                    print(f"{int(i/len(X0) * 100)} % Completado.") 
                    
            X1 = X1 + alpha * (X0[i] - X1) 
            Y1 = Y1 + alpha * (Y0[i] - Y1) 
            X.append(X[i-1] + alpha * (X1 - X[i-1]) )
            Y.append(Y[i-1] + alpha * (Y1 - Y[i-1]) )
        X, Y = X[1:], Y[1:]
    
    return X, Y

def reconstruct_reference_and_quadrature(ts_reference, reference, t0=-0.5, mode="artificial"):  
    fs = 1 / (ts_reference[1] - ts_reference[0])
    Ar, fr = fft_parameters(ts_reference, reference)
    A = np.max([np.max(reference), Ar])
    # print(A, np.max(reference))
    # if (Delta := abs(A r- np.max(reference))/abs(Ar)) > 0.1:
    #     print(f"Maybe amplitud normalization with fft failed, Relative Difference is {Delta*100:.2f} %.")

    if mode == "FDF":
        ts_ref, sin_ref = resample(ts_reference, reference, t0)
        sin_ref /= A 
        cos_ref = shift_90(ts_ref, sin_ref, fr)

    if mode == "artificial":
        ts_ref = ts_reference - t0/fs
        sin_ref = np.sin(2*np.pi*fr * ts_ref)
        cos_ref = np.cos(2*np.pi*fr * ts_ref)

    return sin_ref , cos_ref 

def get_amplitude_and_phase(ts_reference, reference, ts_signal, signal, cut_off=10, ref_mode="FDF", filter_mode="butter", check_progress=False, t0=-0.5):
    sin_ref, cos_ref = reconstruct_reference_and_quadrature(ts_reference, reference, t0,  mode=ref_mode)
    
    X0 = 2 * cos_ref * signal 
    Y0 = 2 * sin_ref * signal

    fs = 1 / (ts_reference[1] - ts_reference[0])
    X, Y = low_pass_filter(X0, Y0, cut_off, fs, filter_mode, check_progress) 

    return np.sqrt(np.array(X)**2 + np.array(Y)**2), np.arctan2(X, Y)       

def lock_in(times, volts, cut_off=10, ref_mode="FDF", filter_mode="butter", plot_results=False, check_progress=False, t0=-0.5):
    t_ref, volts_ref = times[0], volts[0]  # referencia: canal 1
    t_signal, volts_signal = times[1], volts[1]  # señal: canal 2

    A, phi = get_amplitude_and_phase(t_ref, volts_ref, t_signal, volts_signal, cut_off, ref_mode=ref_mode, filter_mode=filter_mode, check_progress=check_progress, t0=t0)

    if ref_mode == "artificial":
        A0, phi0 = get_amplitude_and_phase(t_ref, volts_ref, t_ref, volts_ref, cut_off, ref_mode=ref_mode, filter_mode=filter_mode, check_progress=check_progress, t0=0)
        phi -= phi0

    if plot_results:
        plt.subplot(211)
        plt.plot(t_signal, A) 
        plt.ylabel("Amplitud [V]")
        plt.subplot(212)
        plt.plot(t_signal, phi)  
        plt.ylabel("Fase [rad]")
        plt.xlabel("Tiempo [s]")
        plt.show()

    return A, phi


if __name__ == "__main__":
    t, v = read("Test data/RLC_attenuation.bin", check_metadata=True)  
    A, phi = lock_in(t, v, cut_off=1000, filter_mode="exponential")
    
    plt.plot(t[0][::500], phi[::500])
    plt.show()
