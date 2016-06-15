#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import test0

test0.init(4,5,3,2)
fs = float(150)  # Hz
t = np.arange(2000*fs)/fs

xt = (np.cos(2*np.pi*t) + 3*np.cos(2*np.pi*7*t)).tolist()

for x in xt:
    print(test0.corr_value(np.array([x,x,x,x]),np.array([x,x,x,x,x])))