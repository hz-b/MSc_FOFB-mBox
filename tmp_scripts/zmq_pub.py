
import zmq
ctx = zmq.Context(1)
s = ctx.socket(zmq.PUB)
s.bind("tcp://localhost:3333")
s.send("value")