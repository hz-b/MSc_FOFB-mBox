#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import struct
from zmq_client import *

SAMPLE_NB = 100

def parse_frames(messages):
    sample_nb = len(messages)

    bpm_nbx = np.fromstring(messages[0][2], dtype='double').size
    bpm_nby = np.fromstring(messages[0][3], dtype='double').size

    valuesX = np.zeros((bpm_nbx, sample_nb))
    valuesY = np.zeros((bpm_nby, sample_nb))
    loopPos = []
    # parse frames in values X
    for count, message in enumerate(messages):
        loopPos.append(struct.unpack('i',message[1])[0])
        valuesX[:, count] = np.fromstring(message[2], dtype='double')
        valuesY[:, count] = np.fromstring(message[3], dtype='double')
        
        return valuesX, valuesY, loopPos


if __name__ == "__main__":
#    zclient_curr = ZmqClient()
#    zclient_curr.connect("tcp://gofbz12c.ctl.bessy.de:5563")
#    zclient_curr.subscribe(['FOFB-BPM-DATA'])
#    zclient_curr.subscribe(['FOFB-CM-DATA'])

    zclient_new = ZmqClient()
    zclient_new.connect("tcp://gofbz12c.ctl.bessy.de:3333")
    zclient_new.subscribe(['FOFB-BPM-DATA'])
#    zclient_new.subscribe(['FOFB-CM-DATA'])

    for i in range(SAMPLE_NB):
        # Only to be sure not to lose anything
#        messages_curr = zclient_curr.receive(1)
        messages_new = zclient_new.receive(1)

#        valuesX_curr, valuesY_curr, loopPos_curr = parse_frames(messages_curr)
        valuesX_new, valuesY_new, loopPos_new = parse_frames(messages_new)

#        if loopPos_new == loopPos_curr:
#            if (valuesX_new == valuesX_new).all():
        print('ValuesX correct')
#            else:
#                print('ValuesX incorrect')
        print(valuesX_new.T)
#                print(valuesX_curr.T)                
#            if (valuesY_new == valuesY_new).all():
#                print('ValuesY correct')
#            else:
#                print('ValuesY incorrect')
        print(valuesY_new.T)
#                print(valuesY_curr.T)                

#        elif loopPos_curr[0] == loopPos_new[0] + 1:
#            print("new is late, drop one of curr")
#            messages_curr = zclient_curr.receive(1)
#        elif loopPos_curr[0] == loopPos_new[0] - 1:
#            print("curr is late, drop one of new")
#            messages_new = zclient_new.receive(1)
#        else:
#            print('error: loopPos_curr= {} and loopPos_curr={}'.format(loopPos_curr[0], loopPos_new[0]))

import matplotlib.pyplot as plt

plt.plot(valuesX_new)
plt.plot(valuesY_new)
plt.show()
