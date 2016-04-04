import zmq
s = zmq.Socket(zmq.Context.instance(), zmq.REQ)
s.connect("tcp://localhost:3334")
s.send('KEYLIST')
 
