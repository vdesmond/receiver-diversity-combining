#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def get_bit_error_rate(arr1, arr2, sample_num):
    error_count = np.count_nonzero(arr1!=arr2)
    ber = (error_count/sample_num) * 2
    return ber

def equal_gain(gain, rec_data, sample_num, data):
    rec_egc = np.sum(np.exp( 0-1j * np.angle(gain)) * rec_data, axis = 2)
    rec_egc_real = np.real(rec_egc)
    result_egc = (rec_egc_real > 0).astype(int) * 2 - 1
    
    ber_egc = get_bit_error_rate(result_egc, data, sample_num)
    return ber_egc

def maximal_ratio(gain, rec_data, sample_num, data):
    rec_mrc = np.sum(gain.conjugate() * rec_data, axis=2)
    rec_mrc_real = np.real(rec_mrc)
    result_mrc = (rec_mrc_real > 0).astype(int) * 2 - 1
    
    ber_mrc = get_bit_error_rate(result_mrc, data, sample_num)
    return ber_mrc
