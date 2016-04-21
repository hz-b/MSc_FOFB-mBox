import numpy as np
import struct
import zmq

class ZmqClient:
    subscription_list = []

    def __init__(self, thread_nb=1):
        c = zmq.Context.instance(thread_nb)
        self.socket = zmq.Socket(c, zmq.SUB)

    def connect(self, address):
        self.socket.connect(address)

    def subscribe(self, subscriptions=[""]):
        for title in subscriptions:
            if title not in self.subscription_list:
                self.socket.setsockopt(zmq.SUBSCRIBE, title)
                self.subscription_list.append(title)

    def unsubscribe(self, subscriptions=[]):
        if not len(subscriptions):
            subscriptions = self.subscription_list

        for title in subscriptions:
            if title in self.subscription_list:
                self.socket.setsockopt(zmq.UNSUBSCRIBE, title)
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

class ValuesSubscriber(ZmqClient):
    def __init__(self, thread_nb=1):
        ZmqClient.__init__(self, thread_nb)

    def receive(self, message_nb=1):
        messages = ZmqClient.receive(self, message_nb)

        value_type = messages[0][2];
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
            return valuesX, valuesY, loopPos
        else:
            return valuesX, loopPos
