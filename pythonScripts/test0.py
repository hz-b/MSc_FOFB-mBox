import numpy as np

i = 0

def corr_value(BPMx, BPMy, CMx_nb, CMy_nb):
    global i
    i+=1
    # rate = 150 ms
    """
        CMx then CMy. In this script we don't care about the BPM values
        Loop over each CM, them over each frequency, then time.
        The only way to know where we are in the loop is i. + other global variables
    """
    a = BPMx.size
    b = BPMy.size
    print('{} - {}, {}'.format(i,a,b))
    CMx = np.zeros(CMx_nb)+10
    CMx[20] = 123.45678910
    CMy = np.zeros(CMy_nb)+12
    return CMx, CMy
