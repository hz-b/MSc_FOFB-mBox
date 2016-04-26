import sys
import math
import numpy as np

sys.path.append('search_kicks')
import zmq_client as zc
import search_kicks.tools as sktools

SAMPLE_NB = 100
HOST = 'tcp://gofbz12c:3333'
HOST_REQ = 'tcp://gofbz12c:3334'
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

# Get values
[buff_adc], _ = s_adc.receive(SAMPLE_NB)
sin10 = buff_adc[62,:]
[BPMx, BPMy], _ = s_bpm.receive(SAMPLE_NB)
[CMx, CMy], _ = s_cm.receive(SAMPLE_NB)

ampsin, ampcos = sktools.maths.extract_sin_cos(sin10.reshape(1, SAMPLE_NB), fs=150., f=10., method='fft')
amp10 = np.linalg.norm([ampsin[0], ampcos[0]])
ph10 = math.atan2(ampcos[0], ampsin[0])

# Get all parameters for calculations
sreq = zc.ZmqReq()
sreq.connect(HOST_REQ)

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
asinX, acosX = sktools.maths.extract_sin_cos(BPMx, fs=150., f=10., method='fft')
asinY, acosY = sktools.maths.extract_sin_cos(BPMy, fs=150., f=10., method='fft')
valuesX = acosX + 1j*asinX
valuesY = acosY + 1j*asinY

CorrX = np.dot(Sxx_inv, valuesX)
CorrY = np.dot(Syy_inv, valuesY)

ampX = np.abs(CorrX) / amp10
phX = np.angle(CorrX) - ph10

ampY = np.abs(CorrY) / amp10
phY = np.angle(CorrY) - ph10

if pack.unpack_string(sreq.ask('SET AMPLITUDES-X-10')) == "GO":
    sreq.ask(pack.pack_vec(ampX))
else:
    print("error on amplitudesX")

if pack.unpack_string(sreq.ask('SET PHASES-X-10')) == "GO":
    sreq.ask(pack.pack_vec(phX))
else:
    print("error on phasesX")

if pack.unpack_string(sreq.ask('SET AMPLITUDES-Y-10')) == "GO":
    sreq.ask(pack.pack_vec(ampX))
else:
    print("error on amplitudesY")

if pack.unpack_string(sreq.ask('SET PHASES-Y-10')) == "GO":
    sreq.ask(pack.pack_vec(ampX))
else:
    print("error on phasesY")

