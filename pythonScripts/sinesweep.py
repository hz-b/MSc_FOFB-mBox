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
f_lin = 2*fmin*t+(fmax-fmin)/tmax*0.5*(t**2)
xinput = amplitude*np.sin(2*np.pi*f_lin)

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
    read_bpms((BPMx, BPMy), t_id)

    # Update indexes for next round
    if t_id < sample_nb-1:
        t_id += 1
    else:
        # Save dictionary and erase previous one
        np.save(filename, [res])
        print("saved -- (axis={} / CMB_id={})"
              .format(axis, CM_id))

        # reset time
        t_id = 0

        if axis == 'x':
            if CM_id < gCMx_nb-1:
                CM_id += 1
            else:
                axis = 'y'
                CM_id = 0
        else:
            # it means that we are done with CMx
            if CM_id < gCMy_nb-1:
                CM_id += 1
            else:
                # everything done
                status = Status.Done

    return set_output(axis, CM_id, t_id)


def set_output(axis, CM_id, t_id):
    CMx = np.zeros(gCMx_nb)
    CMy = np.zeros(gCMy_nb)

    if axis == 'x':
        CMx[CM_id] = xinput[t_id]
    else:
        CMy[CM_id] = xinput[t_id]

    return CMx, CMy


def read_bpms(bpms, t_id):
    """  bpms = (BPMx, BPMy), t_id = scalar """

    if axis == 'x':
        res['data']['xx'][:, CM_id, t_id] = bpms[0]
        res['data']['yx'][:, CM_id, t_id] = bpms[1]
    else:
        res['data']['xy'][:, CM_id, t_id] = bpms[0]
        res['data']['yy'][:, CM_id, t_id] = bpms[1]

