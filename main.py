#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main file containing the simulation code along with some plotting functions
Takes in command line arguments
"""

import argparse
import logging
import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from utils import check_modes, direct, equal_gain, maximal_ratio, selective

# ? Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # * Set logging level here (DEBUG or INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)-8s :: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# ? Configure plotting
plt.style.use("seaborn-darkgrid")
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams["figure.figsize"] = 13, 7

# ? For choosing function and in logs
method_dict = {
    "egc": ("Equal Gain", equal_gain),
    "mrc": ("Maximal Ratio", maximal_ratio),
    "dirc": ("Direct", direct),
    "selc": ("Selective", selective),
}

# ? indeplot, combiplot_fading and combiplot_mode are for plotting
def indeplot(ber, mode_n, snr_db_list, fading, no_of_paths):
    """
    This function plots all modes independtly and saves them in given path

    Args:
        BER (ndarray): Array containing BER for all modes
        mode_n (tuple): Sorted tuples of modes to be plotted
        snr_db_list (list): List of SNR(dB) to be plotted against
        fading (str): Type of fading
        no_of_paths (int): Number of branches to be simulated
    """
    for ind, mode_var in enumerate(mode_n):
        plt.plot(snr_db_list, ber[:, :, ind], ":o")
        plt.yscale("log")
        plt.xticks(snr_db_list)

        plt.legend(
            [f"L={l}" for l in range(1, no_of_paths + 1)],
            loc=1,
            frameon=True,
            facecolor="white",
            framealpha=0.8,
        )
        plt.figtext(
            0.5,
            0.9,
            f"{method_dict[mode_var][0]} Combining - R{fading[1:]}",
            fontsize=20,
            ha="center",
        )
        plt.xlabel("SNR (in dB)")
        plt.ylabel("BER")
        plt.savefig(f"./docs/{method_dict[mode_var][0]} Combining - R{fading[1:]}.png")
        plt.clf()


def combiplot_fading(sample_num, no_of_paths, snr_arange, mode):

    """
    This function plots comparision of 2 fading types
    for all given modes and saves them in given path

    Args:
        sample_num (int): Number of samples
        no_of_paths (int): Number of branches to be simulated
        snr_arange (tuple): Arange type input for SNR(dB) to be simulated
        mode (tuple): Modes to be simulated
    """

    ber_rayleigh, mode_n, snr_db_list, _, _ = simulate_combining(
        sample_num, no_of_paths, snr_arange, "rayleigh", mode
    )
    ber_rician, _, _, _, _ = simulate_combining(
        sample_num, no_of_paths, snr_arange, "rician", mode
    )

    for ind, mode_var in enumerate(mode_n):

        plt.plot(snr_db_list, ber_rayleigh[:, :, ind], "--o")
        plt.gca().set_prop_cycle(None)
        plt.plot(snr_db_list, ber_rician[:, :, ind], "-.*")
        plt.yscale("log")
        plt.xticks(snr_db_list)

        lines = plt.gca().get_lines()
        legend1 = plt.legend(
            [lines[i] for i in range(0, no_of_paths)],
            [f"L={l}" for l in range(1, no_of_paths + 1)],
            loc=1,
            frameon=True,
            facecolor="white",
            framealpha=0.8,
        )
        legend2 = plt.legend(
            [lines[i] for i in [0, no_of_paths]],
            ["Rayleigh", "Rician"],
            loc=3,
            frameon=True,
            facecolor="white",
            framealpha=0.8,
        )
        plt.gca().add_artist(legend1)
        plt.gca().add_artist(legend2)

        plt.figtext(
            0.5,
            0.9,
            f"{method_dict[mode_var][0]} Combining - Rayleigh vs Rician",
            fontsize=20,
            ha="center",
        )
        plt.xlabel("SNR (in dB)")
        plt.ylabel("BER")
        plt.savefig(
            f"./docs/{method_dict[mode_var][0]} Combining - Rayleigh vs Rician.png"
        )

        plt.clf()


def combiplot_mode(ber, mode_n, snr_db_list, fading, no_of_paths):

    """
    This function plots comparision of 2 modes for all given modes
    and saves them in given path

    Args:
        BER (ndarray): Array containing BER for all modes
        mode_n (tuple): Sorted tuples of modes to be plotted
        snr_db_list (list): List of SNR(dB) to be plotted against
        fading (str): Type of fading to be simulated
        no_of_paths (int): Number of branches to be simulated
    """
    for mode_combination in list(combinations(mode_n, 2)):
        mode1, mode2 = (
            method_dict[mode_combination[0]],
            method_dict[mode_combination[1]],
        )
        mode1_index, mode2_index = mode_n.index(mode_combination[0]), mode_n.index(
            mode_combination[1]
        )
        plt.plot(snr_db_list, ber[:, 1:, mode1_index], ":o")
        plt.gca().set_prop_cycle(None)
        plt.plot(snr_db_list, ber[:, 1:, mode2_index], "-.*")
        plt.yscale("log")
        plt.xticks(snr_db_list)

        lines = plt.gca().get_lines()
        legend1 = plt.legend(
            [lines[i] for i in range(0, no_of_paths - 1)],
            [f"L={l}" for l in range(2, no_of_paths + 1)],
            loc=1,
            frameon=True,
            facecolor="white",
            framealpha=0.8,
        )
        legend2 = plt.legend(
            [lines[i] for i in [0, no_of_paths - 1]],
            [mode1[0], mode2[0]],
            loc=3,
            frameon=True,
            facecolor="white",
            framealpha=0.8,
        )
        plt.gca().add_artist(legend1)
        plt.gca().add_artist(legend2)

        plt.figtext(
            0.5,
            0.9,
            f"{mode1[0]} vs {mode2[0]} Combining - R{fading[1:]}",
            fontsize=20,
            ha="center",
        )
        plt.xlabel("SNR (in dB)")
        plt.ylabel("BER")
        plt.savefig(f"./docs/{mode1[0]} vs {mode2[0]} Combining - R{fading[1:]}.png")

        plt.clf()


def simulate_combining(sample_num, no_of_paths, snr_arange, fading, mode):

    """
    This function simulates receiver diversity combining for the given arguments

    Args:
        sample_num (int): Number of samples
        no_of_paths (int): Number of branches to be simulated
        snr_arange (tuple): Arange type input for SNR(dB) to be simulated
        fading (str): Type of fading to be simulated
        mode (tuple): Modes to be simulated

    Returns:
        [ndarray]: Array containing BER for all modes
    """
    # ? To calculate simulation time
    start = time.time()

    # ? To generate SNR (dB) based on arange arguments
    snr_db_list = np.arange(*snr_arange)

    # ? To check validity of modes
    mode_n = check_modes(mode)
    if not mode_n:
        logger.error(f"Error in mode: {mode}")
        return -1

    # ? Initialize empty array for storing BER
    ber = np.zeros((len(snr_db_list), no_of_paths, len(mode_n)))

    for snr_index, snr_db in enumerate(snr_db_list):
        snr = 10 ** (snr_db / 10)

        data = np.random.rand(2, sample_num)

        # ? QPSK Modulation
        qpsk_data = 2 * (data > 0.5).astype(int) - 1

        # ? Setting constant Signal Power for transmission
        e_signal = np.sqrt(2)
        e_noise = e_signal / snr

        for path in range(1, no_of_paths + 1):

            # ? Generating noise based on SNR value
            noise = np.random.normal(
                0, np.sqrt(e_noise / 2), size=(2, sample_num, path)
            ) + (
                (0 + 1j)
                * np.random.normal(0, np.sqrt(e_noise / 2), size=(2, sample_num, path))
            )

            # ? Generating channel fading gain
            if fading.lower() == "rayleigh":
                gain = np.random.normal(
                    0, 1 / np.sqrt(2), size=(1, sample_num, path)
                ) + ((0 + 1j) * np.random.normal(0, 1 / 2, size=(1, sample_num, path)))
            elif fading.lower() == "rician":
                gain = np.random.normal(1 / 2, 1 / 2, size=(1, sample_num, path)) + (
                    (0 + 1j)
                    * np.random.normal(
                        1 / np.sqrt(2), 1 / 2, size=(1, sample_num, path)
                    )
                )
            else:
                logger.error(f"{fading} fading channel is not defined.")
                return -1

            # ? Tiling Channel fading gain matrix for QPSK data
            gain_qpsk = np.tile(gain, [2, 1, 1])

            # ? Tiling QPSK data to simulate L branches
            transmitted_signal = np.dstack((qpsk_data,) * path)

            # ? Simulating Received signal
            received_signal = gain_qpsk * transmitted_signal + noise

            for ber_index, method in enumerate(mode_n):

                # ? Calculating BER for given modes
                ber[snr_index, path - 1, ber_index] = method_dict[method][1](
                    gain_qpsk, received_signal, sample_num, qpsk_data
                )
                logger.debug(
                    f"BER = {ber[snr_index, path-1, ber_index]:<10} For Mode ::"
                    f" {method_dict[method][0]:<15} SNR = {snr_db:<5} No of diversity"
                    f" branches = {path:<3} Fading = R{fading[1:]}"
                )

    logger.debug(f"Time taken: {time.time() - start}s")

    return ber, mode_n, snr_db_list, fading, no_of_paths


if __name__ == "__main__":

    # ? Instantiate the argument parser
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=65)
    parser = argparse.ArgumentParser(
        description="Receiver Diversity combining Simulation",
        formatter_class=formatter,
        epilog=(
            "Read the README for more instructions:"
            " https://github.com/Vignesh-Desmond/receiver-diversity-combining/blob/main/README.md"
        ),
    )
    parser.add_argument(
        "-s",
        "--samplenum",
        help=(
            "Specify the number of sample points to be used for the simulation."
            " Default: 100000"
        ),
        type=int,
        default=100000,
    )
    parser.add_argument(
        "-b",
        "--branches",
        help="Number of branches (diversity paths) to be simulated. Default: 3",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-e",
        "--snr",
        help="SNR(dB) to be simulated in numpy arange format. Default: -7 5 2",
        nargs=3,
        metavar=("START_SNR", "END_SNR", "SPACING"),
        type=int,
        default=[-7, 5, 2],
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="Receiver Diversity strategies to be simulated. Default: egc mrc selc",
        nargs="*",
        metavar=("M1", "M2"),
        type=str,
        default=["egc", "mrc", "selc"],
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="Plotting type",
        choices=[
            "default",
            "channel-comparision",
            "mode-comparision",
        ],
        type=str,
        required=True,
    )

    # ? Parse arguments
    args = parser.parse_args()

    SAMPLE_NUM = args.samplenum
    NO_OF_PATHS = args.branches
    SNR_ARANGE = tuple(args.snr)
    MODE = tuple(args.mode)
    PLOT_TYPE = args.plot

    logger.info(
        f"Number of sample points: {SAMPLE_NUM}\n"
        f"Number of branches: {NO_OF_PATHS}\n"
        f"SNR(dB) to be simulated (arange): {SNR_ARANGE}\n"
        f"Diversity strategies: {MODE}\n"
        f"Plotting type: {PLOT_TYPE}\n"
    )

    if SAMPLE_NUM < 1000:
        logger.warning(
            "Low number of sample points (%d), may cause inaccurate simulations",
            SAMPLE_NUM,
        )

    if SNR_ARANGE[1] > 10:
        logger.warning(
            "Simulating for High SNR: %d can cause BER to drop to zero", SNR_ARANGE[1]
        )

    if NO_OF_PATHS > 5:
        logger.warning(
            "No of branches is given as %d. This can cause plots to be cluttered",
            NO_OF_PATHS,
        )

    time.sleep(1)
    logger.info("Starting Simulation")
    time.sleep(1)

    if PLOT_TYPE == "default":
        # * Remove a line if only one channel fading simulation plot is required
        indeplot(
            *simulate_combining(SAMPLE_NUM, NO_OF_PATHS, SNR_ARANGE, "rayleigh", MODE)
        )
        indeplot(
            *simulate_combining(SAMPLE_NUM, NO_OF_PATHS, SNR_ARANGE, "rician", MODE)
        )

    if PLOT_TYPE == "channel-comparision":
        combiplot_fading(SAMPLE_NUM, NO_OF_PATHS, SNR_ARANGE, MODE)

    if PLOT_TYPE == "mode-comparision":
        if len(MODE) < 2:
            logger.error("Mode comparision requires a minimum of 2 modes.")
        else:
            # * Remove a line if only one channel fading simulation plot is required
            combiplot_mode(
                *simulate_combining(
                    SAMPLE_NUM,
                    NO_OF_PATHS,
                    SNR_ARANGE,
                    "rayleigh",
                    MODE,
                )
            )
            combiplot_mode(
                *simulate_combining(
                    SAMPLE_NUM,
                    NO_OF_PATHS,
                    SNR_ARANGE,
                    "rician",
                    MODE,
                )
            )

    logger.info("Simulation completed successfully")
