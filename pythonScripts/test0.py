import numpy as np
import time
#################
# Global values #
#################

# Values
fs = float(150)  # Hz
f_max = 10
f = (np.arange(0, f_max*5)/float(5)).tolist()
t_max = 2
t = (np.arange(t_max*fs)/fs).tolist()
filename = "/home/churlaud/Projects/fofb/mBox++/pythonScripts/results"

# Indexes
axis = 0
CM_id = 0
f_id = 0
t_id = 0

class Status:
    Idle, Run, Done = range(3)

status = Status.Idle


def init(BPMx_nb, BPMy_nb, CMx_nb, CMy_nb):
    """ Initialize the environnment """
    global gBPMx_nb, gBPMy_nb, gCMx_nb, gCMy_nb, status
    global BPMx_buff, BPMy_buff
    global f, filename
    global amp, ph

    gBPMx_nb = BPMx_nb
    gBPMy_nb = BPMy_nb
    gCMx_nb = CMx_nb
    gCMy_nb = CMy_nb

    # Containers
    BPMx_buff = np.zeros((gBPMx_nb, len(t)))
    BPMy_buff = np.zeros((gBPMy_nb, len(t)))
    amp = {'xx': np.zeros((len(f), gBPMx_nb, gCMx_nb)),
           'xy': np.zeros((len(f), gBPMx_nb, gCMy_nb)),
           'yx': np.zeros((len(f), gBPMy_nb, gCMx_nb)),
           'yy': np.zeros((len(f), gBPMy_nb, gCMy_nb))
           }
    ph =  {'xx': np.zeros((len(f), gBPMx_nb, gCMx_nb)),
           'xy': np.zeros((len(f), gBPMx_nb, gCMy_nb)),
           'yx': np.zeros((len(f), gBPMy_nb, gCMx_nb)),
           'yy': np.zeros((len(f), gBPMy_nb, gCMy_nb))
           }
    filename += '_' + time.strftime('%Y-%m-%d_%H-%M-%S')


def corr_value(BPMx, BPMy):
    global gCMx_nb, gCMy_nb, \
           f, f_id, \
           t, t_id, \
           CM_id, axis, \
           status

    if status == Status.Done:
        print('done')
        return set_output(0,0,0,0)

    if status == Status.Idle:
        # Means that it's the first time the function is called
        status = Status.Run
    if status == Status.Run:
        read_bpms((BPMx, BPMy), t_id)

    CM_nb = (gCMx_nb, gCMy_nb)

    if t_id < len(t)-1:
        t_id += 1
    else:
        calc_amp_phase(f_id, CM_id, axis)
        t_id = 0
        if f_id < len(f)-1:
            f_id += 1
        else:
            f_id = 0
            if CM_id < CM_nb[axis]-1:
                CM_id += 1
            else:
                CM_id = 0
                if axis == 0:
                    axis = 1
                else:
                    status = Status.Done

    return set_output(t[t_id], f[f_id], CM_id, axis)


def set_output(t, f, CM_id, axis):
    global gBPMx_nb, gBPMy_nb, gCMx_nb, gCMy_nb
    v = np.sin(2*np.pi*f*t)

    CMx = np.zeros(gCMx_nb)
    CMy = np.zeros(gCMy_nb)

    if axis == 0:
        CMx[CM_id] = v
    else:
        CMy[CM_id] = v

    return CMx, CMy


def read_bpms(bpms, t_id):
    """  bpms = (BPMx, BPMy), t_id = scalar """
    global BPMx_buff, BPMy_buff

    BPMx_buff[:, t_id] = bpms[0]
    BPMy_buff[:, t_id] = bpms[1]


def calc_amp_phase(f_id, CM_id, axis):
    global BPMx_buff, BPMy_buff
    global f, fs
    global amp, ph

    fftx = np.fft.fft(BPMx_buff, axis=1)*2/BPMx_buff.shape[1]
    ffty = np.fft.fft(BPMy_buff, axis=1)*2/BPMy_buff.shape[1]

    freqs = np.fft.fftfreq(fftx.shape[1],1/fs)

    idx = np.argmin(np.abs(freqs-f[f_id]))

    ampx = np.absolute(fftx[:,idx])
    ampy = np.absolute(ffty[:,idx])
    phx = np.angle(fftx[:,idx])
    phy = np.angle(ffty[:,idx])

    if axis == 0:
        amp['xx'][f_id, :, CM_id] = ampx
        amp['yx'][f_id, :, CM_id] = ampy
        ph['xx'][f_id, :, CM_id] = phx
        ph['yx'][f_id, :, CM_id] = phy
    else:
        amp['xy'][f_id, :, CM_id] = ampx
        amp['yy'][f_id, :, CM_id] = ampy
        ph['xy'][f_id, :, CM_id] = phx
        ph['yy'][f_id, :, CM_id] = phy

    save_to_file()


def save_to_file():
    """ Save the 2 dictionaries: delete previous one."""
    global amp, ph, f
    global filename

    np.save(filename, [{'amplitudes': amp, 'phases': ph, 'freqs': f}])
    print('save')
