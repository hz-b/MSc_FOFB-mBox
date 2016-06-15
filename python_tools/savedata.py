#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import time
import numpy as np
from zmq_client import ValuesSubscriber
#from PyML import PyML

SAMPLE_NB = 100
EVERYX = 70

def show():

#    pml.ao[Family]['Status'][

    s = ValuesSubscriber()
    s.connect("tcp://localhost:3333")
    s.subscribe(['FOFB-BPM-DATA',
#                 'FOFB-CM-DATA'
                 ])
#    fig = plt.figure(figsize=(10,5))
#    f1 = fig.add_subplot(2,1,1)
#    f2 = fig.add_subplot(2,1,2)
#    plt.ion()
#    plt.show()
    t = 0
    # Only to be sure not to lose anything
    [valuesX, valuesY], loopPos = s.receive(1000)

    np.save('tenHz' + '_' + time.strftime('%Y-%m-%d_%H-%M-%S'), [valuesX, valuesY])

if __name__ == "__main__":
    show()
