import zmq_client as zc
import numpy as np

s = zc.ValuesSubscriber()
s.connect('tcp://localhost:3333')
s.subscribe(['FOFB-ADC-DATA'])
buff,loop= s.receive(1)
print(buff)
