from __future__ import division, print_function

import numpy as np
import time

#################
# Global values #
#################

# Values
fs = float(150)  # Hz

fmin = 0
fmax = fs/2
tmax = 50
t = np.arange(tmax*fs)/fs
sample_nb = t.size
filename = "sine_sweep"

# input
amplitude = 0.02
f = 6

def xinput_fct():
    sample = 0
    while True:
        yield amplitude*np.sin(2*np.pi*fi*sample/fs)
        sample += 1

xinput = xinput_fct()

# Indexes
axis = 'x'
CM_id = 0
t_id = 0

res = None
gBPMx_nb, gBPMy_nb, gCMx_nb, gCMy_nb = 0, 0, 0, 0


class Status:
    Idle, Run, Done = range(3)


status = Status.Idle


def init(BPMx_nb, BPMy_nb, CMx_nb, CMy_nb):
    """ Initialize the environnment """
    global filename, gBPMx_nb, gBPMy_nb, gCMx_nb, gCMy_nb, res

    gBPMx_nb = BPMx_nb
    gBPMy_nb = BPMy_nb
    gCMx_nb = CMx_nb
    gCMy_nb = CMy_nb

    res = {'data': {'xx': np.zeros((gBPMx_nb, gCMx_nb, sample_nb)),
                    'xy': np.zeros((gBPMx_nb, gCMy_nb, sample_nb)),
                    'yx': np.zeros((gBPMy_nb, gCMx_nb, sample_nb)),
                    'yy': np.zeros((gBPMy_nb, gCMy_nb, sample_nb))
                    },
           'shape': "d['data']['xy'][2, 4, 31] = 31th time sample of 2nd BPMx "
                    "when 4th CMy active.",
           'input': xinput,
           'amplitude': amplitude,
           'date': time.strftime('%Y-%m-%d %H:%M:%S')
           }

    filename += '_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.npy'
    np.save(filename, [res])

    print("\033[95m [Python] Saving in {}\033[00m".format(filename))


def corr_value(BPMx, BPMy):
    global status, t_id, axis, CM_id

    if status == Status.Done:
        print('done')
        return np.zeros(gCMx_nb), np.zeros(gCMy_nb)

    if status == Status.Idle:
        # Means that it's the first time the function is called
        status = Status.Run
        print("Let's start")
        return set_output(axis, CM_id, t_id)

    # Those are the result of last round
    read_bpms((BPMx, BPMy), axis, CM_id, t_id)

    # Update indexes for next round
    if t_id < sample_nb-1:
        t_id += 1

    return set_output(axis, CM_id, t_id)


def set_output(axis, CM_id, t_id):
    CMx = np.zeros(gCMx_nb)
    CMy = np.zeros(gCMy_nb)

    if axis == 'x':
        CMx[CM_id] = xinput.next()
    else:
        CMy[CM_id] = xinput.next()

    return CMx, CMy


def read_bpms(bpms, axis, CM_id,  t_id):
    """  bpms = (BPMx, BPMy), t_id = scalar """

    res['data']['x'+axis][:, CM_id, t_id] = bpms[0]
    res['data']['y'+axis][:, CM_id, t_id] = bpms[1]

