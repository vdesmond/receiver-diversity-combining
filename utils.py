#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np 

def get_ber(A, B, sample_num):
    # ! possible error
    error_count = np.count_nonzero(A==B)
    Pe = error_count/sample_num/2
    return Pe

def equal_gain(g2, r, sample_num, data):
    # ! possible error found ya ig (nvm)

    temp0 = np.angle(g2)
    temp1 = np.exp( -1j * temp0)

    # ! bodha
    temp2 = temp1 * r
    # ! ^ bodha ends

    temp3 = np.sum(temp2, axis = 2)
    r_egc = np.real(temp3)
    # print(r.mean(axis=0).mean(axis=0).mean(axis=0))
    # print(r_egc.mean(axis=0).mean(axis=0))
    result = (r_egc > 0).astype(int) * 2 - 1
    ber = get_ber(result, data, sample_num)
    return [ber, result]