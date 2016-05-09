import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 1:
    print('filename as argument')
    exit()
bpm = None
if len(sys.argv) > 2:
    bpm = int(sys.argv[2])

filename = sys.argv[1]

data = np.load(filename)[0]
freqs = data['freqs']
amps = data['amplitudes']
ph = data['phases']

#print abs(amps['xx'][:,:,0])
plt.figure()
plt.subplot(2,1,1)

if bpm is None:
    plt.plot(freqs, 20*np.log10(abs(amps['xx'][:,:,0])))
else:
    plt.plot(freqs, 20*np.log10(abs(amps['xx'][:,bpm,0])))

plt.subplot(2,1,2)
if bpm is None:
    plt.plot(freqs, ph['xx'][:,:,0])
else:
    plt.plot(freqs, ph['xx'][:,bpm,0])


plt.show()
