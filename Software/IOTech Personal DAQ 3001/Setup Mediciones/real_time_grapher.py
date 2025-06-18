import numpy as np
import time
import os
from PyIOTech import daq, daqh
from Formatter import get_converted_data
from lock_in import get_amplitude_and_phase  # Tu implementación

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys

# === Inicialización de PyQtGraph ===
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(title="Fase en tiempo real")
plot = win.addPlot(title="Fase (última muestra por bloque)")
curve = plot.plot(pen='y', symbol='o')
plot.addLine(y=0, pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
plot.setLabel('left', "Fase", units='rad')
plot.setLabel('bottom', "Tiempo", units='s')
win.show()

# === Parámetros graficador === #
max_dots = 500
mean_length = 100 

# === Configuración DAQ ===
device_name = b'PersonalDaq3001{374679}'
dev = daq.daqDevice(device_name)

flags = daqh.DafAnalog | daqh.DafBipolar | daqh.DafDifferential | daqh.DafSettle1us
gain = daqh.DgainX1
freq = int(500000)  # Hz
buf_size = 8192
n_channels = 2

dev.AdcSetScan([0, 1], [gain] * n_channels, [flags] * n_channels)
dev.AdcSetFreq(freq)
actual_freq = dev.AdcGetFreq()
print(f"Frecuencia real de muestreo por canal: {actual_freq:.1f} Hz")

dev.AdcSetAcq(daqh.DaamInfinitePost, 0, 0)
dev.AdcSetTrig(daqh.DatsSoftware, 0, 0, 0, 0)
dev.AdcTransferSetBuffer(daqh.DatmUpdateBlock | daqh.DatmCycleOn | daqh.DatmIgnoreOverruns, buf_size)

# === Arranque ===
dev.AdcArm()
dev.AdcTransferStart()
dev.AdcSoftTrig()

print("Adquisición comenzada...")

# === Loop de adquisición infinita ===
try:
    phases = []
    timestamps = []
    t_start = time.time()

    while True:
        t0 = time.time()
        dev.WaitForEvent(daqh.DteAdcData)
        status = dev.AdcTransferGetStat()

        if status['retCount'] < buf_size:
            continue

        binary_data = dev.dataBuf
        times_arr, voltages_arr = get_converted_data(binary_data, actual_freq, n_channels)

        ts_ref, ref = times_arr[0], voltages_arr[0]
        ts_sig, sig = times_arr[1], voltages_arr[1]

        amp, phase = get_amplitude_and_phase(ts_ref, ref, ts_sig, sig, cut_off=1005)

        if len(phases) >= max_dots:
            phases.pop(0)
            timestamps.pop(0)

        phases.append(np.mean(phase[-mean_length:]))
        timestamps.append(time.time() - t_start)

        # === Actualizar gráfico ===
        curve.setData(timestamps, phases)
        QtWidgets.QApplication.processEvents()

except KeyboardInterrupt:
    print("Adquisición interrumpida por el usuario.")

finally:
    dev.AdcDisarm()
    dev.Close()
    print("Dispositivo cerrado.")
