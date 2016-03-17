import zmq

ctx = zmq.Context(1)
s = ctx.socket(zmq.SUB)
s.connect("tcp://localhost:3333")
s.setsockopt(zmq.SUBSCRIBE, "VALUE")

while True:
    first_frame = True
    while True:
        message = s.recv(copy=False)
        if first_frame:
            first_frame = False
        else:
            print(message)
        if not message.more:
            break
