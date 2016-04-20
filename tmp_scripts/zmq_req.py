import zmq
import struct
import numpy as np

def unpack_double(v):
    return struct.unpack('d', v)[0]

def unpack_int(v):
    return struct.unpack('i', v)[0]

def unpack_string(v):
    return str(v)

def unpack_vec(v):
    return np.fromstring(v, dtype=np.double)

def unpack_mat(v, dims):
    return np.fromstring(v, dtype=np.double).reshape(dims)

def pack_double(v):
    return struct.pack('d', v)

def pack_int(v):
    return struct.pack('i', v)

def pack_string(v):
    return str(v)

def pack_vec(v):
    return np.tostring(v)

def pack_mat(v, dims):
    return np.tostring(v)

s = zmq.Socket(zmq.Context.instance(), zmq.REQ)
s.connect("tcp://localhost:3334")
s.send('KEYLIST')
print(s.recv())
