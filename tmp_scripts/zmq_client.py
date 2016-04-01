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

