# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:14:15 2025

@author: Alex
"""

import sys
sys.path.append('D:\\Codigos_py\\Repositorio')

from Fourier_CSPD import np, mfft, mcfft
import pyLINT as lint
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis, skew

#%%

path_LINT_levcol = ['D:/Proyecto LINT at Prompt Gamma/2024-11-06_pequeñasvarPG/datos_06.11.24/15_19_6nov24_levantecolimador.log']
path_SEAD_levcol = 'D:/Proyecto LINT at Prompt Gamma/2024-11-06_pequeñasvarPG/datos_06.11.24/Datos RA-3/15_19_6nov24_levantecolimador_SEAD.TXT'

t_inicio_col = pd.Timestamp('2024-11-06 15:19:28', tz=None) #Set tiempo inicial (SEAD)
data_LINT_col = lint.data_LINT(path_LINT_levcol, t_inicio_SEAD = t_inicio_col, dt = -40)
data_LINT_col.corr_rep()

dt = data_LINT_col.time_cond[0][-1] - data_LINT_col.time_cond[0][0]
print('freq. de sampleo fue de aprox.', len(data_LINT_col.time_cond[0])/(60*dt), 'adq/s')

datos_sead_col = lint.data_sead(path_SEAD_levcol, delim = ',')
plot_SEAD_col, ax_SEAD_col = lint.figure_SEAD(datos_sead_col, ['CI N16-1'], idx_list = [''] , yscale='linear')

N_t = 20
data_sel = ['time_norep_hhmmss', 'counts_norep'] #en hh:mm:ss, cps #['time_norep', 'counts_norep'] en s, cps

fig_col, ax_col = lint.fig_LINT_attr(data_LINT_col, data_sel)
fig_col.show()
t_fix = pd.date_range(f'{data_LINT_col.time_norep_hhmmss[0][0]}', f'{data_LINT_col.time_norep_hhmmss[0][-1]}', periods = N_t)
ax_col.set_xticks(ticks = np.linspace(0, 60*data_LINT_col.time_norep[0][-1], num=N_t), labels = [str(tt)[0:8] for tt in t_fix.time])
fig_col.autofmt_xdate()

ax_col.set_ylabel('CPS')
ax_col.set_xlabel('t [hh:mm:ss]')

#%%

#levante del colimador
nombres = ['CI N16-1', 'CI N16-2']
y_SEAD = datos_sead_col.CIC_Marcha('CI N16-1', '').to_numpy() [69:569]
y_LINT =  datos_sead_col.CIC_Marcha('CI N16-2', '').to_numpy()[69:569] #data_LINT_col.counts_norep[0][:500]
#datos_sead_col.CIC_Marcha('M4', '').to_numpy()[69:569]#

at = 1
FS = 1/at
L = 2**7#1024

NAPSD = np.zeros((2, L))
PSD = np.zeros((2, L))
db = np.zeros((2, L))

for i, x in enumerate([y_LINT, y_SEAD]):
    NAPSD[i, :], f, PSD[i, :], db[i, :] = mfft(at, x, blksize=2**8)
 
plt.figure()
for i in range(2):
    plt.plot(f, NAPSD[i], label = nombres[i])
plt.yscale('log')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel ('NAPSD')
plt.legend()
plt.grid()

plt.figure()
for i in range(2):
    plt.plot(f, db[i, :], label = nombres[i])
plt.xlabel('Frecuencia (Hz)')
plt.ylabel ('PSD [dB]')
plt.legend()
plt.grid()

cpsd, coh, phase, f = mcfft(1, y_SEAD, y_LINT, blksize=2**8)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(f, coh.T)
ax1.set_ylabel('Coherencia')
ax1.grid()
ax2.plot(f, phase.T, 'o-')
ax2.grid()
ax2.set_xlabel('Frecuencia [Hz]')
ax2.set_ylabel('Fase [Grados]')
fig.suptitle('Coherencia y fase entre Detectores {} y {}'.format(*nombres)) 
fig.tight_layout()
