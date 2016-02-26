
import zmq
ctx = zmq.Context(1)
s = ctx.socket(zmq.SUB)
s.connect("tcp://localhost:3333")
s.setsockopt(zmq.SUBSCRIBE, "")

while True:
    print(s.recv())
