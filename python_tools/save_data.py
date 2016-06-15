#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys
import numpy as np
import matplotlib.pyplot as plt
from zmq_client import ValuesSubscriber
from PyML import PyML

SAMPLE_NB = 1

if __name__=='__main__':
    if len(sys.argv) > 1:
        filename = str(sys.argv[1])
        print(filename)
    else:
        filename = 'savenow'

    pml = PyML()
    pml.setao(pml.loadFromExtern('PyML/bessyIIinit.py','ao'))

#    pml.ao[Family]['Status'][
    posBPMx = pml.getfamilydata('BPMx', 'Pos')[pml.getActiveIdx('BPMx')]
    posBPMy = pml.getfamilydata('BPMy', 'Pos')[pml.getActiveIdx('BPMy')]
    posCMx = pml.getfamilydata('HCM', 'Pos')[pml.getActiveIdx('HCM')]
    posCMy = pml.getfamilydata('VCM', 'Pos')[pml.getActiveIdx('VCM')]

    sCM = ValuesSubscriber()
    sCM.connect("tcp://localhost:3333")
    sCM.subscribe(['FOFB-CM-DATA' ])
    # Only to be sure not to lose anything
    sBPM = ValuesSubscriber()
    sBPM.connect("tcp://localhost:3333")
    sBPM.subscribe(['FOFB-BPM-DATA' ])
    # Only to be sure not to lose anything
    bx = []
    by = []
    cx = []
    cy = []
    while 1:
        try:
            [valuesX, valuesY], loopPos = sBPM.receive(1)
            bx.append(valuesX[:,0].tolist())
            by.append(valuesY[:,0].tolist())
            [valuesX, valuesY], loopPos = sCM.receive(1)
            cx.append(valuesX[:,0].tolist())
            cy.append(valuesY[:,0].tolist())
        except:
            d = dict()
            d['BPM'] = {'x': np.array(bx).T, 'y': np.array(by).T}
            d['CM'] = {'x': np.array(cx).T, 'y': np.array(cy).T}
            np.save(filename,[d])
            print('--------------------------------------')
            print('--------------DATA SAVED--------------')
            print('--------------------------------------')
            raise
