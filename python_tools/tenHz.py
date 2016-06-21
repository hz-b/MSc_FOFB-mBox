from __future__ import division, print_function, unicode_literals

import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
sys.path.append('search_kicks')
sys.path.append('../../../search_kicks')
import zmq_client as zc
import search_kicks.tools as sktools


def fit_coefs(values, asin, acos, fs, f):
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

        return amps, ampc

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
    SAMPLE_NB = 100

    s_bpm, s_cm, s_adc, sreq = init()
    # Get values
    (buff_adc), _ = s_adc.receive(SAMPLE_NB)
    sin10 = buff_adc[62,:]
    (BPMx, BPMy), _ = s_bpm.receive(SAMPLE_NB)
    (CMx, CMy), _ = s_cm.receive(SAMPLE_NB)

    acos, asin = sktools.maths.extract_sin_cos(sin10.reshape(1, SAMPLE_NB), fs=150., f=10.)
    ampc, amps = fit_coefs(sin10.reshape(1, SAMPLE_NB), acos, asin, fs=150, f=10)
    amp10 = np.linalg.norm([amps[0], ampc[0]])
    ph10 = math.atan2(amps[0], ampc[0])

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
    asinX, acosX = sktools.maths.extract_sin_cos(BPMx, fs=150., f=10.)
    asinX, acosX = fit_coefs(BPMx, acosX, asinX, fs=150, f=10)
    asinY, acosY = sktools.maths.extract_sin_cos(BPMy, fs=150., f=10.)
    asinY, acosY = fit_coefs(BPMy, acosY, asinY, fs=150, f=10)

    valuesX = acosX + 1j*asinX
    valuesY = acosY + 1j*asinY

    CorrX = np.dot(Sxx_inv, valuesX)
    CorrY = np.dot(Syy_inv, valuesY)

    ampX = np.abs(CorrX)
    phX = np.angle(CorrX)

    ampY = np.abs(CorrY)
    phY = np.angle(CorrY)

    plt.figure()
    t = np.arange(SAMPLE_NB)/150
    plt.plot(t, sin10)
    plt.plot(t, np.cos(2*np.pi*10*t+ph10))
    plt.show()

    ans = pack.unpack_string(sreq.tell('SET AMPLITUDE-REF-10',
                                       pack.pack_double(amp10)))
    if ans != "ACK":
        print("error on AMPLITUDE-REF-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET PHASE-REF-10', pack.pack_double(ph10)))
    if ans != "ACK":
        print("error on SET PHASE-REF-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET AMPLITUDES-X-10', pack.pack_vec(ampX)))
    if ans != "ACK":
        print("error on AMPLITUDES-X-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET PHASES-X-10', pack.pack_vec(phX)))
    if ans != "ACK":
        print("error on PHASES-X-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET AMPLITUDES-Y-10', pack.pack_vec(ampY)))
    if ans != "ACK":
        print("error on AMPLITUDES-Y-10: {}".format(ans))

    ans = pack.unpack_string(sreq.tell('SET PHASES-Y-10', pack.pack_vec(phY)))
    if ans != "ACK":
        print("error on PHASES-Y-10: {}".format(ans))

    print('ampX')
    print(ampX)
    print('ampY')
    print(ampY)
    print('phX')
    print(phX)
    print('phY')
    print(phY)
