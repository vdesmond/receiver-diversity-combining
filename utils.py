#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np 

def get_ber(A, B, sample_num):
    # * error wasnt even here
    error_count = np.count_nonzero(A==B)
    Pe = error_count/sample_num/2
    return Pe

def equal_gain(g2, r, sample_num, data):
    r_egc = np.real(np.sum(np.multiply(np.exp( 0-1j * np.angle(g2)), r), axis = 2))
    result = (r_egc > 0).astype(int) * 2 - 1
    ber = get_ber(result, data, sample_num)
    return [ber, result]