import signal
import sys
import zmq

COLORS = True
KEYS = [
#        "VALUE",
        "LOG",
        "ERROR"
        ]

"""
    FRAME LOG/ERROR = header | time | message | other (optional)
"""

def read_str(s):
    return str(s)[:-1]  # We don't want the \0 at the end.

class Log:
    def __init__(self):
        self.header = str()
        self.time = str
        self.message = str()
        self.other = str()

    def output(self, color):
        if color:
            if self.header == "ERROR":
                color_in = "\x1b[1;31m"
            if self.header == "LOG":
                color_in = "\x1b[1;36m"
            color_out = "\x1b[0m"
        else:
            color_in = ''
            color_out = ''

        value = self.time + ' ' \
                + color_in + '[' + self.header + ']' + color_out + ' ' \
                + self.message

        if self.other:
            value += ' [' + self.other + ']'

        return value

def signal_handler(signal, frame):
    print('\nCtrl+C pressed, terminating...')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    use_file = False
    color = True
    if len(sys.argv) == 3:
        if sys.argv[1] == "-o":
            color = False
            use_file = True
            fileoutput = str(sys.argv[2])

    ctx = zmq.Context(1)
    s = ctx.socket(zmq.SUB)
    s.connect("tcp://localhost:3333")
    for key in KEYS:
        s.setsockopt(zmq.SUBSCRIBE, key)

    while True:
        count = 0
        log = Log()
        while True:  # Frame loop
            message = s.recv(copy=False)

            if count == 0:
                log.header = read_str(message)

            if log.header == "LOG" or log.header == "ERROR":
                if count == 1:
                    log.time = read_str(message)
                elif count == 2:
                    log.message = read_str(message)
                elif count == 3:
                    log.other = read_str(message)

            if not message.more:
                if log.header == "LOG" or log.header == "ERROR":
                    if use_file:
                        f = open(fileoutput, 'a')
                        f.write(log.output(color) + '\n')
                        f.close()
                    else:
                        print(log.output(color))
            count += 1

