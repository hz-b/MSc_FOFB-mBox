#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short description.

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""
from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("ticks")

plt.close('all')
try:
    a = np.load('/home/churlaud/with_corr_dyncorr_ampx200.npy', encoding='bytes')[0]
    b = np.load('/home/churlaud/with_dyncorr_only_ampx200.npy', encoding='bytes')[0]
    c = np.load('/home/churlaud/withoutcorr.npy', encoding='bytes')[0]
except:
    a = np.load('/home/churlaud/with_corr_dyncorr_ampx200.npy')[0]
    b = np.load('/home/churlaud/with_dyncorr_only_ampx200.npy')[0]
    c = np.load('/home/churlaud/withoutcorr.npy')[0]
titles = ['Dynamic + normal corr', 'Dynamic corr and P=0', 'No correction']


def plotfft(x, typeof, axis, i):
    N = x[typeof][axis].shape[1]
    freqs = np.fft.fftfreq(N, 1/150)[:N//2]
    X = np.fft.fft(x[typeof][axis][i, :])[:N//2]*2/N*100
    X[0] = 0
    plt.plot(freqs, abs(X))
    plt.title(typeof + axis)
    plt.xlabel('Frequency [in Hz]')
    plt.ylabel('Relative amplitude')
    sns.despine()
    plt.grid()


for i, x in enumerate([a, b, c]):
    plt.figure(titles[i])
    plt.subplot(221)
    plotfft(x, 'BPM', 'x', 0)
    plt.ylim(ymax=0.25)
    plt.subplot(223)
    plotfft(x, 'BPM', 'y', 0)
    plt.ylim(ymax=0.20)
    plt.subplot(222)
    plotfft(x, 'CM', 'x', 0)
    plt.subplot(224)
    plotfft(x, 'CM', 'y', 0)
    plt.tight_layout()

plt.figure()
plt.subplot(211)
plotfft(c, 'BPM', 'x', 0)
sns.despine()
plt.subplot(212)
plotfft(c, 'BPM', 'y', 0)
sns.despine()

plt.tight_layout()

plt.show()