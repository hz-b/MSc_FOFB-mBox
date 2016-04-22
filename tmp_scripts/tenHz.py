import zmq_client as zc
import numpy as np
import matplotlib.pyplot as plt

s = zc.ValuesSubscriber()
s.connect('tcp://localhost:3333')
s.subscribe(['FOFB-ADC-DATA'])
buff,loop = s.receive(1000)
plt.plot(np.fft.fft(buff[62,:]))
plt.show()
