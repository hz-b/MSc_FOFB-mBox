#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from zmq_client import ValuesSubscriber
from PyML import PyML

SAMPLE_NB = 100


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
    plt.ion()
    plt.show()
    p1X = None
    p2X = None
    t = 0
    while True:
        # Only to be sure not to lose anything
        valuesX, valuesY, loopPos = s.receive(1)
        if t < 100:
            t += 1
            continue
        if valuesX.size > 64: # it's a bpm value
            f1 = fig.add_subplot(2,1,1)
            if p1X is not None:
                p1X.pop(0).remove()
                p1Y.pop(0).remove()
            p1X = plt.plot(posBPMx, valuesX[:,0], '-b')
            p1Y = plt.plot(posBPMy, valuesY[:,0], '-g')
        else:
            f2 = plt.subplot(2,1,2)
            t = 0
            if p2X is not None:
                p2X.pop(0).remove()
                p2Y.pop(0).remove()
            p2X = plt.plot(posCMx, valuesX[:,0], '-b')
            p2Y = plt.plot(posCMy,valuesY[:,0], '-g')
        plt.draw()
        print(loopPos[0])

def compare():
    zclient_curr = ValuesSubscriber()
    zclient_curr.connect("tcp://gofbz12c.ctl.bessy.de:5563")
    zclient_curr.subscribe(['FOFB-BPM-DATA'])
    zclient_curr.subscribe(['FOFB-CM-DATA'])

    zclient_new = ValuesSubscriber()
    zclient_new.connect("tcp://gofbz12c.ctl.bessy.de:3333")
    zclient_new.subscribe(['FOFB-BPM-DATA'])
    zclient_new.subscribe(['FOFB-CM-DATA'])

    for i in range(SAMPLE_NB):
        # Only to be sure not to lose anything
        valuesX_curr, valuesY_curr, loopPos_curr = zclient_curr.receive(1)
        valuesX_new, valuesY_new, loopPos_new = zclient_new.receive(1)

        if loopPos_new == loopPos_curr:
            if (valuesX_new == valuesX_new).all():
                print('ValuesX correct')
            else:
                print('ValuesX incorrect')
                print(valuesX_new.T)
                print(valuesX_curr.T)
            if (valuesY_new == valuesY_new).all():
                print('ValuesY correct')
            else:
                print('ValuesY incorrect')
                print(valuesY_new.T)
                print(valuesY_curr.T)

        elif loopPos_curr[0] == loopPos_new[0] + 1:
            print("new is late, drop one of curr")
            messages_curr = zclient_curr.receive(1)
        elif loopPos_curr[0] == loopPos_new[0] - 1:
            print("curr is late, drop one of new")
            messages_new = zclient_new.receive(1)
        else:
            print('error: loopPos_curr= {} and loopPos_curr={}'
                .format(loopPos_curr[0], loopPos_new[0]))

if __name__ == "__main__":
    #compare()
    show()
