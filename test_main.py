#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import *
from temp_utils import *
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_NUM = 10000
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

            noise = load_mat(L, SNR_index, n=True)

            if fading == "Rayleigh":
                gain = load_mat(L, SNR_index, n=False)
            # ? dont care about rician for now
            elif fading == "Rician":
                gain = np.random.normal(1/2, 1/2, size = (1,sample_num,L)) + \
                    ((0+1j)*np.random.normal(1/np.sqrt(2), 1/2 ,size = (1,sample_num,L)))
            else:
                print("No fading channel type given.")
                return -1
            
            # print(noise.mean(axis=0).mean(axis=0).mean(axis=0))
            gain_qpsk = np.tile(gain,[2,1,1])

            transmitted_signal = np.dstack((qpsk_data, ) * L)
            received_signal = gain_qpsk * transmitted_signal + noise
            # print(transmitted_signal.mean(axis=0).mean(axis=0))

            Pe[SNR_index, L-1], _ = equal_gain(gain_qpsk, received_signal, sample_num, qpsk_data)
    
    plt.plot(SNR_dB_list, Pe)
    plt.legend()
    plt.show()

simulate_combining()