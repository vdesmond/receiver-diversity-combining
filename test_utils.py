#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.io
import numpy as np

PATH = "/home/desmond/Desktop/Diversity_Combining/"
def load_mat(L, SNR_index, name="n"):
    if name=="n":
        mat = scipy.io.loadmat(PATH+f'{SNR_index+1}_{L}_n.mat')['n']
    elif name=="g":
        mat = scipy.io.loadmat(PATH+f'{SNR_index+1}_{L}_g.mat')['g']
    elif name=="r":
        mat = scipy.io.loadmat(PATH+f'{SNR_index+1}_{L}_r.mat')['r']
    elif name=="q":
        mat = scipy.io.loadmat(PATH+f'{SNR_index+1}_{L}_q.mat')['g_tmp']
    elif name=="t":
        mat = scipy.io.loadmat(PATH+f'{SNR_index+1}_{L}_t.mat')['tx_data']
    elif name=="d":
        mat = scipy.io.loadmat(PATH+'qpsk.mat')['data']
    else:
        print("Error")
        return -1
    if L==1:
        if name != "d":
            mat = np.expand_dims(mat, axis=2)
    return mat 