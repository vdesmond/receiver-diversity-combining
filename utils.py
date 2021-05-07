#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility file containing the methods for receiver diversity combining
along with some helper methods
"""
from itertools import chain, combinations

import numpy as np


def equal_gain(gain, rec_data, sample_num, data):
    """
    This function calculates BER for Equal Gain Combining

    Args:
        gain (ndarray): Channel Fading Gain matrix of dimensions (2, sample_num, L)
        rec_data (ndarray): Receiver data matrix of dimensions (2, sample_num, L)
        sample_num (int): Number of samples
        data (ndarray): QPSK data matrix of dimensions (2, sample_num, L)

    Returns:
        [float]: Bit Error Rate for Equal Gain Combining
    """
    rec_egc = np.sum(np.exp(0 - 1j * np.angle(gain)) * rec_data, axis=2)
    rec_egc_real = np.real(rec_egc)
    result_egc = qpsk_detection(rec_egc_real)

    ber_egc = get_bit_error_rate(result_egc, data, sample_num)
    return ber_egc


def maximal_ratio(gain, rec_data, sample_num, data):
    """
    This function calculates BER for Maximal Ratio Combining

    Args:
        gain (ndarray): Channel Fading Gain matrix of dimensions (2, sample_num, L)
        rec_data (ndarray): Receiver data matrix of dimensions (2, sample_num, L)
        sample_num (int): Number of samples
        data (ndarray): QPSK data matrix of dimensions (2, sample_num, L)

    Returns:
        [float]: Bit Error Rate for Maximal Ratio Combining
    """
    rec_mrc = np.sum(gain.conjugate() * rec_data, axis=2)
    rec_mrc_real = np.real(rec_mrc)
    result_mrc = qpsk_detection(rec_mrc_real)

    ber_mrc = get_bit_error_rate(result_mrc, data, sample_num)
    return ber_mrc


def direct(gain_qpsk, rec_data, sample_num, data):
    """
    This function calculates BER for Direct Combining

    Args:
        gain (ndarray): Placeholer argument as gain matrix is not needed
        rec_data (ndarray): Receiver data matrix of dimensions (2, sample_num, L)
        sample_num (int): Number of samples
        data (ndarray): QPSK data matrix of dimensions (2, sample_num, L)

    Returns:
        [float]: Bit Error Rate for Direct Combining
    """
    # ? channel fading is not needed here, the argument is just a placeholder.
    rec_dir = np.sum(rec_data, axis=2)
    rec_dir_real = np.real(rec_dir)
    result_dir = qpsk_detection(rec_dir_real)

    ber_dir = get_bit_error_rate(result_dir, data, sample_num)
    return ber_dir


def selective(gain_qpsk, rec_data, sample_num, data):
    """
    This function calculates BER for Selective Combining

    Args:
        gain (ndarray): Channel Fading Gain matrix of dimensions (2, sample_num, L)
        rec_data (ndarray): Receiver data matrix of dimensions (2, sample_num, L)
        sample_num (int): Number of samples
        data (ndarray): QPSK data matrix of dimensions (2, sample_num, L)

    Returns:
        [float]: Bit Error Rate for Selective Combining
    """
    max_gain_index = np.argmax(np.absolute(gain_qpsk[1:]), axis=2)

    rec_data_temp = np.zeros((2, sample_num), dtype="complex_")
    gain_qpsk_temp = np.zeros((2, sample_num), dtype="complex_")

    for i in range(sample_num):
        rec_data_temp[:, i] = rec_data[:, i, int(max_gain_index[:, i])]
        gain_qpsk_temp[:, i] = gain_qpsk[:, i, int(max_gain_index[:, i])]

    rec_sel = np.exp(0 - 1j * np.angle(gain_qpsk_temp)) * rec_data_temp  # ? im stupid
    rec_sel_real = np.real(rec_sel)
    result_sel = qpsk_detection(rec_sel_real)

    ber_sel = get_bit_error_rate(result_sel, data, sample_num)
    return ber_sel


def qpsk_detection(data):
    """
    This function performs QPSK demodulation on given data

    Args:
        data (ndarray): Modulated QPSK data

    Returns:
        [ndarray]: Demodulated data
    """
    return (data > 0).astype(int) * 2 - 1


def get_bit_error_rate(arr1, arr2, sample_num):
    """
    This function calculates Bit Error Rate by comparing 2 matrices
    which are assumed as received data (after QPSK Demodulation) and
    original data before transmission

    Args:
        arr1 (ndarray): Numpy array
        arr2 (ndarray): Numpy array
        sample_num (int): Number of samples

    Returns:
        [float]: Bit Error Rate
    """
    error_count = np.count_nonzero(arr1 != arr2)
    ber = (error_count / sample_num) * 2
    return ber


def check_modes(mode):
    """
    This function checks the mode tuple and returns a sorted tuple
    if all modes are valid.

    Args:
        mode (tuple): Tuple of modes to be simulated

    Returns:
        [tuple]: Sorted tuple if modes are valid else None
    """
    sorted_mode = tuple(sorted(mode))
    all_modes = ("dirc", "egc", "mrc", "selc")
    all_combs = list(
        chain(*(list(combinations(all_modes, i + 1)) for i in range(len(all_modes))))
    )
    if sorted_mode in all_combs:
        return sorted_mode
