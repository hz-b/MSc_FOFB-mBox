from __future__ import division, print_function, unicode_literals

import numpy as np
import struct
import zmq


class ZmqSubscriber:
    subscription_list = []

    def __init__(self, thread_nb=1):
        c = zmq.Context.instance(thread_nb)
        self.socket = zmq.Socket(c, zmq.SUB)

    def connect(self, address):
        self.socket.connect(address)

    def subscribe(self, subscriptions=[""]):
        for title in subscriptions:
            if title not in self.subscription_list:
                self.socket.setsockopt_string(zmq.SUBSCRIBE, title)
                self.subscription_list.append(title)

    def unsubscribe(self, subscriptions=[]):
        if not len(subscriptions):
            subscriptions = self.subscription_list

        for title in subscriptions:
            if title in self.subscription_list:
                self.socket.setsockopt_string(zmq.UNSUBSCRIBE, title)
                self.subscription_list.remove(title)
            else:
                print("ERROR: I'm not subscribed to '{}'".format(title))

    def receive(self, message_nb=1):
        messages = []
        for i in range(message_nb):
            is_more = True
            message = []
            while is_more:
                frame = self.socket.recv(copy=False)
                message.append(frame.bytes)
                is_more = frame.more
            messages.append(message)

        return messages


class ValuesSubscriber(ZmqSubscriber):
    def __init__(self, thread_nb=1):
        ZmqSubscriber.__init__(self, thread_nb)

    def receive(self, message_nb=1):
        messages = ZmqSubscriber.receive(self, message_nb)

        value_type = messages[0][2]
        valx_nb = np.fromstring(messages[0][3], dtype=value_type).size
        valuesX = np.zeros((valx_nb, message_nb))

        if len(messages[0]) > 4:
            valy_nb = np.fromstring(messages[0][4], dtype=value_type).size
            valuesY = np.zeros((valy_nb, message_nb))
        loopPos = []

        # parse frames in values X
        for count, message in enumerate(messages):
            loopPos.append(struct.unpack('i', message[1])[0])
            valuesX[:, count] = np.fromstring(message[3], dtype=value_type)
            if len(messages[0]) > 4:
                valuesY[:, count] = np.fromstring(message[4], dtype=value_type)

        if len(messages[0]) > 4:
            return [valuesX, valuesY], loopPos
        else:
            return [valuesX], loopPos


class ZmqReq:
    def __init__(self, thread_nb=1):
        c = zmq.Context.instance(thread_nb)
        self.socket = zmq.Socket(c, zmq.REQ)

    def connect(self, address):
        self.socket.connect(address)

    def ask(self, query):
        self.socket.send_string(query)
        return self.socket.recv()

    def tell(self, query, val):
        self.socket.send_string(query, zmq.SNDMORE)
        self.socket.send(val)
        return self.socket.recv()

class Packer:
    import struct
    def unpack_double(self, v):
        return struct.unpack('d', v)[0]

    def unpack_int(self, v):
        return struct.unpack('i', v)[0]

    def unpack_string(self, v):
        return str(v)

    def unpack_vec(self, v):
        return np.fromstring(v, dtype=np.double)

    def unpack_mat(self, v, dims):
        return np.fromstring(v, dtype=np.double).reshape(dims)

    def pack_double(self, v):
        return struct.pack('d', v)

    def pack_int(self, v):
        return struct.pack('i', v)

    def pack_string(self, v):
        return str(v)

    def pack_vec(self, v):
        return v.tostring()

    def pack_mat(self, v):
        return v.tostring()

