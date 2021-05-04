#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.io
import numpy as np

PATH = "/home/desmond/Desktop/Wireless_Communication_System-HW3-Diversity_Combining/"
def load_mat(L, SNR_index, n=True):
    if n:
        mat = scipy.io.loadmat(PATH+f'{SNR_index+1}_{L}_n.mat')['n']
    else:
        mat = scipy.io.loadmat(PATH+f'{SNR_index+1}_{L}_g.mat')['g']
    if L ==1:
        mat = np.expand_dims(mat, axis=2)
    return mat