# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:38:13 2025

@author: Alex Canchanya Acosta
"""
import sys
sys.path.append('D:\\Codigos_py\\Repositorio')

from Fourier_CSPD import mfft, mcfft, np
import scipy as sci
import matplotlib.pyplot as plt

path = 'D:/Bibliografía/Curso_Ruido_16al20Dic24/MATLABL/CPSDs/archivos_mat/datos.mat'
data = sci.io.loadmat(path, struct_as_record=False, squeeze_me=True)
datos = data['datos']
at = 0.02
FS = 1/at
L = 1024
N, n = np.shape(datos)

NAPSD = np.zeros((n, L))
PSD = np.zeros((n, L))
db = np.zeros((n, L))

for i in range(n):
    NAPSD[i, :], f, PSD[i, :], db[i, :] = mfft(at, datos[:, i])
 
plt.figure()
for i in range(n):
    plt.plot(f, NAPSD[i])
plt.yscale('log')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel ('NAPSD')
plt.grid()

i1, i2 = 1, 3 
cpsd, coh, phase, f = mcfft(at, datos[:,i1], datos[:, i2])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(f, coh.T)
ax1.set_ylabel('Coherencia')
ax1.grid()
ax2.plot(f, phase.T)
ax2.grid()
ax2.set_xlabel('Frecuencia [Hz]')
ax2.set_ylabel('Fase [Grados]')
fig.suptitle(f'Espectros entre Detectores {i1} y {i2}') 
