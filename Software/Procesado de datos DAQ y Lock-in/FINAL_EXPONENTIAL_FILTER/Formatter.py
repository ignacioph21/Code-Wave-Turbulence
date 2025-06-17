import matplotlib.pyplot as plt
import numpy as np
import json
import os

def convert(binary_data, max_voltage = 10.0, bit_depth = 16):
    return np.array(binary_data) * max_voltage * 2 / (2 ** bit_depth) - max_voltage

def read(filename, max_voltage = 10.0, bit_depth = 16, plot_values=False, check_metadata=False): # TODO: MAX VOLTAGE Y BIT DEPTH DE METADATA.
    metadata_path = os.path.splitext(filename)[0] + "_metadata.json"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    sample_frequency = metadata.get("frecuencia_Hz", 1_000_000)  
    channels = metadata.get("canales", [0]) # TODO: Cambiar nombre de las cosas de metadata.
    scans = metadata.get("scan Count", "Unknwon") # TODO: ARREGLAR EN INTERFAZ. 
    n_channels = len(channels)

    binary_data = np.fromfile(filename, dtype=np.uint16)

    if check_metadata:
        if scans != (saved_scans := len(binary_data)//n_channels):
            print(f"Warning! The number of scans transferred by the DAQ ({scans}) does not match the number of scans saved to disk ({saved_scans}). Maybe a buffer overrun ocurred.")
        print(f"The file '{filename}' contains {saved_scans} scans, each for {n_channels} channels, sampled at {sample_frequency} Hz per channel.")

    voltages = convert(binary_data, max_voltage, bit_depth)

    channels_data = [voltages[i::n_channels] for i in range(n_channels)]
    n_samples = len(channels_data[0])

    times = [np.arange(n_samples) / sample_frequency + i * 1e-6 for i in range(n_channels)] # Habría que sumar según el valor de DafSttle(x)us.

    times_arr = np.vstack(times)
    voltages_arr = np.vstack(channels_data)

    if plot_values:
        for i, (ts, vs) in enumerate(zip(times_arr, voltages_arr)):
            plt.plot(ts, vs, ".-", label=f"Canal {i+1}.")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Voltaje [V]")
        plt.legend()
        plt.show()

    return times_arr, voltages_arr


