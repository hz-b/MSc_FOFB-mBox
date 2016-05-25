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

def frequency_int():
    while True:
        time.sleep(1/fs)
        cbox.set_interruption('adc')

if __name__ == "__main__":
    cbox.set_filename('../build/dump_rmf.dat')
    t = threading.Thread(target=frequency_int)
    t.start()
