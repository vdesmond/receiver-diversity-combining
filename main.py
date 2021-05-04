#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import logging

# ? Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # * Set logging level here (DEBUG or INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)-8s :: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

SAMPLE_NUM = 100000
NO_OF_PATHS = 6
SNR_ARANGE = (-5, 5, 2)
FADING="Rician"
TYPE="egc"

def simulate_combining(sample_num=SAMPLE_NUM, no_of_paths=NO_OF_PATHS, snr_arange=SNR_ARANGE, fading=FADING, type=TYPE):
   
    SNR_dB_list = np.arange(*snr_arange)
    Pe = np.zeros((len(SNR_dB_list),no_of_paths))
    
    for SNR_index, SNR_dB in enumerate(SNR_dB_list):
        SNR = 10 ** (SNR_dB / 10)

        data = np.random.rand(2, sample_num)
        qpsk_data = 2 * (data > 0.5).astype(int) -1

        E_signal = np.sqrt(2)
        E_noise = E_signal/ SNR 
        
        for L in range(1,no_of_paths+1):

            noise = np.random.normal(0,np.sqrt(E_noise/2), size = (2,sample_num,L)) + \
                    ((0+1j)*np.random.normal(0,np.sqrt(E_noise/2),size = (2,sample_num,L)))
           

            if fading.lower() == "rayleigh":
                gain = np.random.normal(0, 1/np.sqrt(2), size = (1,sample_num,L)) + \
                    ((0+1j)*np.random.normal(0 ,1/2 ,size = (1,sample_num,L)))
            elif fading.lower() == "rician":
                gain = np.random.normal(1/2, 1/2, size = (1,sample_num,L)) + \
                    ((0+1j)*np.random.normal(1/np.sqrt(2), 1/2 ,size = (1,sample_num,L)))
            else:
                logger.error(f"{fading} fading channel is not defined.")
                return -1
            
            gain_qpsk = np.tile(gain,[2,1,1])

            transmitted_signal = np.dstack((qpsk_data, ) * L)
            received_signal = gain_qpsk * transmitted_signal + noise

            Pe[SNR_index, L-1], _ = equal_gain(gain_qpsk, received_signal, sample_num, qpsk_data)
            logger.debug(f"BER = {Pe[SNR_index, L-1]:<10} For SNR = {SNR_index} and No of diversity branches = {L}")
    
    plt.plot(SNR_dB_list, Pe)
    plt.yscale("log")
    plt.xticks(SNR_dB_list)
    plt.legend([f"L={l}" for l in range(1,no_of_paths+1)])
    plt.show()

if __name__ =="__main__":
    simulate_combining()