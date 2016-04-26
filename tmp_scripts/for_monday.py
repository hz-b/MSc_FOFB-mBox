#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from zmq_client import ValuesSubscriber
from PyML import PyML

SAMPLE_NB = 100
EVERYX = 70

def show():
    pml = PyML()
    pml.setao(pml.loadFromExtern('PyML/bessyIIinit.py','ao'))

#    pml.ao[Family]['Status'][
    posBPMx = pml.getfamilydata('BPMx', 'Pos')[pml.getActiveIdx('BPMx')]
    posBPMy = pml.getfamilydata('BPMy', 'Pos')[pml.getActiveIdx('BPMy')]
    posCMx = pml.getfamilydata('HCM', 'Pos')[pml.getActiveIdx('HCM')]
    posCMy = pml.getfamilydata('VCM', 'Pos')[pml.getActiveIdx('VCM')]

    s = ValuesSubscriber()
    s.connect("tcp://localhost:5563")
    s.connect("tcp://localhost:3333")
    s.subscribe(['FOFB-BPM-DATA',
                 'FOFB-CM-DATA'
                 ])
    fig = plt.figure(figsize=(10,5))
    f1 = fig.add_subplot(2,1,1)
    f2 = fig.add_subplot(2,1,2)
    plt.ion()
    plt.show()
    t = 0
    while True:
        # Only to be sure not to lose anything
        [valuesX, valuesY], loopPos = s.receive(1)

        if t < EVERYX:
            t += 1
            continue
        if valuesX.size > 64: # it's a bpm value
            f1.clear()
            f1.plot(posBPMx, valuesX[:,0], '-g')
            f1.plot(posBPMy, valuesY[:,0], '-b')
            f1.autoscale()
        else:
            t = 0
            f2.clear()
            f2.plot(posCMx, valuesX[:,0], '-g')
            f2.plot(posCMy, valuesY[:,0], '-b')
            f2.autoscale()
        plt.draw()
        print(loopPos[0])

if __name__ == "__main__":
    show()
