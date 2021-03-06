#! /usr/bin/env python
from __future__ import division, print_function, unicode_literals

import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
sys.path.insert(0,'../../../search_kicks')
import zmq_client as zc
import search_kicks.tools as sktools


def fit_coefs(values, acos, asin, fs, f):
    def func(t, a, b, c, f):
        return a + b*np.cos(2*np.pi*f*t)+c*np.sin(2*np.pi*f*t)

    M, N = values.shape
    t = np.arange(N)/fs
    offs = np.zeros(M)
    ampc = np.zeros(M)
    amps = np.zeros(M)
    freq = np.zeros(M)
    for idx in range(M):
        y = values[idx, :]
        res, _ \
            = optimize.curve_fit(func, t, y,
                                 [np.mean(y), acos[idx], asin[idx], f])

        [offs[idx], ampc[idx], amps[idx], freq[idx]] = res

        return ampc, amps

def init():
    #HOST = 'tcp://gofbz12c:3333'
    #HOST_REQ = 'tcp://gofbz12c:3334'
    HOST = 'tcp://localhost:3333'
    HOST_REQ = 'tcp://localhost:3334'

    s_adc = zc.ValuesSubscriber()
    s_adc.connect(HOST)
    s_adc.subscribe(['FOFB-ADC-DATA'])

    s_cm = zc.ValuesSubscriber()
    s_cm.connect(HOST)
    s_cm.subscribe(['FOFB-CM-DATA'])

    s_bpm = zc.ValuesSubscriber()
    s_bpm.connect(HOST)
    s_bpm.subscribe(['FOFB-BPM-DATA'])

    # Verify that all values are in the same loopPos
    synchronized = False
    while not synchronized:
        _, loop_adc = s_adc.receive()
        _, loop_bpm = s_bpm.receive()
        _, loop_cm = s_cm.receive()

        if loop_adc == loop_bpm and loop_adc == loop_cm:
            synchronized = True
        else:
            print("{} {} {}".format(loop_adc, loop_bpm, loop_cm))

    sreq = zc.ZmqReq()
    sreq.connect(HOST_REQ)

    return s_bpm, s_cm, s_adc, sreq

if __name__=="__main__":
    SAMPLE_NB = 510

    s_bpm, s_cm, s_adc, sreq = init()
    # Get values
    (buff_adc), _ = s_adc.receive(SAMPLE_NB)
    sin10 = buff_adc[62,:]
    (BPMx, BPMy), _ = s_bpm.receive(SAMPLE_NB)
    (CMx, CMy), _ = s_cm.receive(SAMPLE_NB)

    amp10, ph10 = sktools.maths.extract_sin_cos(sin10.reshape(1, SAMPLE_NB), 150., 10., 'polar')
 #   ampc, amps = fit_coefs(sin10.reshape(1, SAMPLE_NB), acos, asin, fs=150, f=10)

    # Get all parameters for calculations
    pack = zc.Packer()
    CMx_nb = pack.unpack_int(sreq.ask('GET NB-CM-X'))
    CMy_nb = pack.unpack_int(sreq.ask('GET NB-CM-Y'))
    BPMx_nb = pack.unpack_int(sreq.ask('GET NB-BPM-X'))
    BPMy_nb = pack.unpack_int(sreq.ask('GET NB-BPM-Y'))
    Sxx = pack.unpack_mat(sreq.ask('GET SMAT-X'), (BPMx_nb, CMx_nb))
    Syy = pack.unpack_mat(sreq.ask('GET SMAT-Y'), (BPMy_nb, CMy_nb))
    ivecX = pack.unpack_double(sreq.ask('GET IVEC-X'))
    ivecY = pack.unpack_double(sreq.ask('GET IVEC-Y'))

    Sxx_inv = sktools.maths.inverse_with_svd(Sxx, ivecX)
    Syy_inv = sktools.maths.inverse_with_svd(Syy, ivecY)

    # Do calculations
    aX, pX = sktools.maths.extract_sin_cos(BPMx, 150., 10., 'polar')
    aY, pY = sktools.maths.extract_sin_cos(BPMy, 150., 10., 'polar')

    valuesX = aX*np.exp(1j*pX)
    valuesY = aY*np.exp(1j*pY)

    CorrX = np.dot(Sxx_inv, valuesX)
    CorrY = np.dot(Syy_inv, valuesY)

    ampX = np.abs(CorrX)
    phX = np.angle(CorrX)

    ampY = np.abs(CorrY)
    phY = np.angle(CorrY)

#    plt.figure()
#    t = np.arange(SAMPLE_NB)/150
#    plt.plot(t, sin10)
#    plt.plot(t, amp10*np.cos(2*np.pi*10*t+ph10))
#    plt.plot(t, ampc*np.cos(2*np.pi*10*t)+amps*np.sin(2*np.pi*10*t))
#    plt.plot(t, acos*np.cos(2*np.pi*10*t)+asin*np.sin(2*np.pi*10*t))
#    plt.show()

    ans = pack.unpack_string(sreq.tell('SET AMPLITUDE-REF-10',
                                       pack.pack_double(amp10)))
    if ans != "ACK":
        print("error on AMPLITUDE-REF-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET PHASE-REF-10', pack.pack_double(ph10)))
    if ans != "ACK":
        print("error on SET PHASE-REF-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET AMPLITUDES-X-10', pack.pack_vec(0.01*ampX)))
    if ans != "ACK":
        print("error on AMPLITUDES-X-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET PHASES-X-10', pack.pack_vec(phX)))
    if ans != "ACK":
        print("error on PHASES-X-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET AMPLITUDES-Y-10', pack.pack_vec(0.01*ampY)))
    if ans != "ACK":
        print("error on AMPLITUDES-Y-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET PHASES-Y-10', pack.pack_vec(phY)))
    if ans != "ACK":
        print("error on PHASES-Y-10: {}".format(ans))

   # print('ampX')
   # print(ampX)
    print('ampY')
    print(ampY)
   # print('phX')
   # print(phX)
    print('phY')
    print(phY)
    print('amp10')
    print(amp10)
    print('ph10')
    print(ph10)
  #  plt.figure()
  #  plt.plot(ampX*np.cos(phX))
  #  plt.plot(ampX*np.sin(phX))
  #  plt.figure()
  #  plt.plot(acosX)
  #  plt.plot(asinX)
  #  plt.show()

