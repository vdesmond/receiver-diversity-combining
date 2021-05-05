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


method_dict = {"egc":("Equal Gain", equal_gain),"mrc":("Maximal Ratio", maximal_ratio), "dir":("Direct",direct)}

SAMPLE_NUM = 100000
NO_OF_PATHS = 6
SNR_ARANGE = (-5, 5, 2)
FADING="Rician"
MODE=("egc","mrc","dir")

def simulate_combining(sample_num=SAMPLE_NUM, no_of_paths=NO_OF_PATHS, snr_arange=SNR_ARANGE, fading=FADING, mode=MODE):
   
    SNR_dB_list = np.arange(*snr_arange)

    mode_n =  check_modes(mode)
    if not mode_n:
        logger.error(f"Error in mode: {mode}")
        return -1

    BER = np.zeros((len(SNR_dB_list),no_of_paths,len(mode_n)))
    
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

            for BER_index, method in enumerate(mode_n):
                BER[SNR_index, L-1, BER_index] = method_dict[method][1](gain_qpsk, received_signal, sample_num, qpsk_data)
                logger.debug(f"BER = {BER[SNR_index, L-1, BER_index]:<10} For Mode :: {method_dict[method][0]:<15} SNR = {SNR_index:<5} No of diversity branches = {L}")

    
    for index in range(len(mode_n)):
        plt.plot(SNR_dB_list, BER[:,:,index])
        plt.yscale("log")
        plt.xticks(SNR_dB_list)
        plt.legend([f"L={l}" for l in range(1,no_of_paths+1)])
        plt.show()

if __name__ =="__main__":
    simulate_combining()