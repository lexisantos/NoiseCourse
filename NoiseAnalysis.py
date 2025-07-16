# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:44:29 2024

@author: Alex
"""

import scipy as sci
import matplotlib.pyplot as plt
import numpy as np

data = sci.io.loadmat('D:/Bibliografía/Curso_Ruido_16al20Dic24/MATLABL/ANALISIS_FOURIER/datos/medidas1_1.mat',
                      struct_as_record=False, squeeze_me=True)

datos = data['medidas1_1']

sample_rate = datos.FS
X = datos.registros[0].datos

f, t, Sxx = sci.signal.spectrogram(X, sample_rate)

plt.pcolormesh(t, f, Sxx, shading = 'gouraud')
plt.ylabel('Frecuency [Hz]')
plt.xlabel('Time [s]')

# N=1024//2
# X1 = np.array([sum(x) for x in np.array_split(X, len(X)//N)])
# f, Pxx = sci.signal.periodogram(X1, sample_rate, window = sci.signal.windows.hamming(N), nfft = N)
f, Pxx = sci.signal.periodogram(X, sample_rate, window = sci.signal.windows.hamming(len(X)), nfft = len(X), detrend = False)
plt.semilogy(f, Pxx)
plt.xlabel('Frecuency [Hz]')
plt.ylabel('PSD [V^2/Hz]')

#%%  Abrir con analisislint_escalones
pow_sel = '2M'
X1 = current_M3_step[pow_sel]
X = data_LINT.counts_norep[pow_sel]

f, Pxx = sci.signal.periodogram((X - X.mean())/X.std(ddof=1), fs=1.0, window = sci.signal.windows.hamming(len(X)), nfft = len(X), detrend = False)
f1, Pxx1 = sci.signal.periodogram((X1 - X1.mean())/X1.std(ddof=1), fs=1.0, window = sci.signal.windows.hamming(len(X1)), nfft = len(X1), detrend = False)

plt.figure()
plt.semilogy(f1, Pxx1, label = 'M3')
plt.semilogy(f, Pxx, label = 'LINT')

plt.xlabel('Frecuency [Hz]')
plt.ylabel('PSD [V^2/Hz]')
plt.legend()
plt.title(pow_sel)

plt.figure()
plt.csd(Pxx, Pxx1)