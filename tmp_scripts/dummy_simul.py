#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short description.

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""

from __future__ import division, print_function

import time
import threading

import cbox

fs = 150

def frequency_int(which, T):
    while True:
        if cbox.is_interruption_enabled(which):
            if T == 0:
                T = 0.1
            time.sleep(T)
            cbox.set_interruption(which)


if __name__ == "__main__":
    cbox.set_filename('../build/dump_rmf.dat')
    tadc = threading.Thread(target=frequency_int, args=['adc', 1/fs] )
    tdac = threading.Thread(target=frequency_int, args=['dac', 0.])
    tadc.start()
    tdac.start()
