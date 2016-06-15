#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short description.

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""
from __future__ import division, print_function, unicode_literals

import sys

import matplotlib.pyplot as plt
import numpy as np

import search_kicks.tools as sktools

#import seaborn as sns

#sns.set_style("ticks")

if len(sys.argv) > 1:
    files = sys.argv[1:]

plt.close('all')

orbits = []
for filename in files:
    orbits.append(sktools.io.load_orbit(filename))

for i, x in enumerate(orbits):
    x.plot_fft(0, files[i])

plt.show()
