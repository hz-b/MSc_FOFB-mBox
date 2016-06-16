#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys
import datetime

import numpy as np
import matplotlib.pyplot as plt

from zmq_client import ValuesSubscriber
from PyML import PyML
import search_kicks.tools as sktools

SAMPLE_NB = 1

if __name__=='__main__':
    measure_date = datetime.datetime.now()
    if len(sys.argv) > 1:
        filename = str(sys.argv[1])
    else:
        strtime = measure_date.isoformat('_'.encode('ascii'))
        strtime = '-'.join(strtime.split(':'))
        filename = 'dump-save-' + strtime
    print("Save in {}".format(filename))

    pml = PyML()
    pml.setao(pml.loadFromExtern('PyML/bessyIIinit.py', 'ao'))

    posBPMx = pml.getfamilydata('BPMx', 'Pos')[pml.getActiveIdx('BPMx')]
    posBPMy = pml.getfamilydata('BPMy', 'Pos')[pml.getActiveIdx('BPMy')]
    posCMx = pml.getfamilydata('HCM', 'Pos')[pml.getActiveIdx('HCM')]
    posCMy = pml.getfamilydata('VCM', 'Pos')[pml.getActiveIdx('VCM')]

    sCM = ValuesSubscriber()
    sCM.connect("tcp://localhost:3333")
    sCM.subscribe(['FOFB-CM-DATA'])

    sBPM = ValuesSubscriber()
    sBPM.connect("tcp://localhost:3333")
    sBPM.subscribe(['FOFB-BPM-DATA'])

    bx = []
    by = []
    cx = []
    cy = []
    i = 0
    while 1:
        try:
            i += 1
            if i % (150*5) == 0:  # Every 5s
                print("Elapsed time = {}s".format(i/150))
            [valuesX, valuesY], loopPos = sBPM.receive(1)
            bx.append(valuesX[:, 0].tolist())
            by.append(valuesY[:, 0].tolist())
            [valuesX, valuesY], loopPos = sCM.receive(1)
            cx.append(valuesX[:, 0].tolist())
            cy.append(valuesY[:, 0].tolist())
        except KeyboardInterrupt:
            orbit = sktools.io.OrbitData(BPMx=np.array(bx).T,
                                         BPMy=np.array(by).T,
                                         CMx=np.array(cx).T,
                                         CMy=np.array(cy).T,
                                         names=None,
                                         sampling_frequency=150,
                                         measure_date=measure_date)
            sktools.io.save_orbit_hdf5(filename, orbit)
            print('--------------------------------------')
            print('--------------DATA SAVED--------------')
            print('--------------------------------------')
            sys.exit(0)
        except Exception:
            raise
