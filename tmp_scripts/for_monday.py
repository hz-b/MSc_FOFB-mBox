#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from zmq_client import ValuesSubscriber

SAMPLE_NB = 100


def show():
    s = ValuesSubscriber()
    s.connect("tcp://localhost:3333")
    s.subscribe(['FOFB-BPM-DATA',
                 'FOFB-CM-DATA'
                 ])
    while True:
        # Only to be sure not to lose anything
        valuesX, valuesY, loopPos = s.receive(1)
        print(loopPos[0])
        print(valuesX[:,0])
        print(valuesY[:,0])

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
