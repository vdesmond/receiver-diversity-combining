#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

SAMPLE_NUM = 100000
NO_OF_PATHS = 4
SNR_INDEX_LIMIT = 5
FADING="Rayleigh"
TYPE="egc"

def simulate_combining(sample_num=SAMPLE_NUM, no_of_paths=NO_OF_PATHS, snr_index_limit=SNR_INDEX_LIMIT, fading=FADING, type=TYPE):

    Pe = np.zeros((snr_index_limit,no_of_paths))
    SNR_dB_list = [(SNR_index-1)*2+1 for SNR_index in range(1, snr_index_limit+1)]
    for SNR_index, SNR_dB in enumerate(SNR_dB_list):
        SNR = 10 ** (SNR_dB / 10)

        data = np.random.rand(2, sample_num)
        qpsk_data = 2 * (data > 0.5).astype(int) -1

        E_signal = np.sqrt(2)
        E_noise = E_signal/ SNR 
        
        for L in range(1,no_of_paths+1):

            noise = np.random.normal(0,np.sqrt(E_noise/2), size = (2,sample_num,L)) + \
                ((0+1j)*np.random.normal(0,np.sqrt(E_noise/2),size = (2,sample_num,L)))

            if fading == "Rayleigh":
                gain = np.random.normal(0, 1/np.sqrt(2), size = (1,sample_num,L)) + \
                    ((0+1j)*np.random.normal(0 ,1/2 ,size = (1,sample_num,L)))
            elif fading == "Rician":
                gain = np.random.normal(1/2, 1/2, size = (1,sample_num,L)) + \
                    ((0+1j)*np.random.normal(1/np.sqrt(2), 1/2 ,size = (1,sample_num,L)))
            else:
                print("No fading channel type given.")
                return -1
            
            gain_qpsk = np.tile(gain,[2,1,1])

            # ! possible error
            transmitted_signal = np.dstack((data, ) * L)
            received_signal = gain_qpsk * transmitted_signal + noise

            if SNR_index == 0 and L == 4:
                print(received_signal)
            Pe[SNR_index, L-1], _ = equal_gain(gain_qpsk, received_signal, sample_num, qpsk_data)
    
    plt.plot(SNR_dB_list, Pe)
    plt.show()

simulate_combining()