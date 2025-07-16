# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:08:40 2025

@author: Alex Canchanya Acosta
"""
import numpy as np 
from scipy.fft import fft
from scipy.stats import skew, kurtosis

def momentos_est(data):
    est = np.zeros(5)
    est[0] = np.mean(data)
    est[1] = np.std(data, ddof = 1)
    est[2] = np.std(data, ddof = 1)**2
    est[3] = skew(data)
    est[4] = kurtosis(data, fisher = False)
    return est

def mfft(at, X1, blksize: int=4096//2):
    '''
    Calcula la PSD normalizada (NAPSD) y la PSD en dB, para un único vector de datos.

    Parameters
    ----------
    at : TYPE
        DESCRIPTION.
    X1 : TYPE
        DESCRIPTION.
    blksize : int, optional
        DESCRIPTION. The default is 4096//2.

    Returns
    -------
    NAPSD : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    psd : TYPE
        DESCRIPTION.
    db : TYPE
        DESCRIPTION.

    '''
    DC = np.mean(X1)
    data = X1 - DC
    
    nblk = int(len(X1)/blksize)
    
    Fs = 1/at
    # Fny = 0.5*Fs
    psd = np.zeros(int(blksize/2))
    
    aidx = np.arange(1, int(blksize/2 + 1))
    f = aidx/blksize*Fs
    
    for i in range(1, nblk+1):
        idx = np.arange((i-1)*blksize, i*blksize, dtype = int)
        fac = fft(data[[idx]]).reshape(-1)
        psd += np.real(fac[:blksize//2]*fac[:blksize//2].T.conj())
    
    psd = psd/nblk
    db = 10*np.log10(psd/psd[4])
    
    NAPSD = psd/(0.5*Fs*blksize*DC**2)
    return NAPSD, f, psd, db


def mcfft(at, datos1, datos2, blksize: int=4096//2):
    DC = np.array([np.mean(datos1), np.mean(datos2)])
    X1 = (datos1 - DC[0])/np.std(datos1, ddof = 1)
    X2 = (datos2 - DC[1])/np.std(datos2, ddof = 1)
    
    data = np.array([X1, X2])
    nblk = int(len(X1)/blksize)
    
    Fs = 1/at
    
    psd = np.zeros((2, int(blksize/2)))
    cpsd = np.zeros((int(blksize/2)), dtype = 'complex_')
    
    aidx = np.arange(1, int(blksize/2 + 1))
    f = aidx/blksize*Fs
    
    for i in range(1, nblk+1):
        idx = np.arange((i-1)*blksize, i*blksize, dtype = int)
        for k in range(2):
            fac = fft(data[k][[idx]]).reshape(-1)
            psd[k, :] += np.real(fac[:blksize//2]*fac[:blksize//2].T.conj())
    
    NAPSD = np.zeros(psd.shape)
    db = np.zeros(psd.shape)
    
    for k in range(2):
        psd[k, :] = psd[k, :]/nblk
        db[k, :] = 10*np.log10(psd[k, :])/psd[k, 4]
        NAPSD[k, :] =  psd[k,:]/(0.5*Fs*blksize*DC[k]**2)
    
    for i in range(1, nblk+1):
        idx = np.arange((i-1)*blksize, i*blksize, dtype = int)
        fac1 = fft(data[0][[idx]]).reshape(-1)
        fac2 = fft(data[1][[idx]]).reshape(-1)
        cpsd += fac1[:blksize//2]*fac2[:blksize//2].T.conj()
    
    cpsd = cpsd/nblk
    
    coh = np.abs(cpsd*cpsd.T.conj())/(psd[0, :]*psd[1, :])
    phase = np.angle(cpsd.T.conj(), deg=True)
    
    return cpsd, coh, phase, f