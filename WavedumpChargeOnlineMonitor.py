# -*- coding:utf-8 -*-
# @Time: 2024/4/2 10:53
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: WavedumpChargeOnlineMonitor.py
# Learn from robjfoster's online method
import matplotlib.pylab as plt
import numpy as np

import sys

import argparse
import sys
import os
import time
from matplotlib.animation import FuncAnimation
from glob import glob

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from gimmedatwave.gimmedatwave import gimmedatwave as gdw  # nopep8

# Set Global Variables
global bins_charge
global bin_centers
global dict_bin_counts
global error_counts
global dict_parser
global time_tolerant_no_event # seconds



dict_parser = {}
dict_bin_counts = {}
dict_hist_patches = {}

error_count = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", help="input directory contained *.dat")
    parser.add_argument('--digitizer', choices=[member.name for member in gdw.DigitizerFamily],
                        help='Select a digitizer family', required=True)

    # Binning Strategy for Charge Spectrum
    parser.add_argument('--bin-start', type=float, default=-1000,help="Start of the charge binning")
    parser.add_argument('--bin-end', type=float, default=10000,help="End of the charge binning")
    parser.add_argument('--nbins', type=int, default=400,help="Number of bins for the charge histogram")

    # Time Tolerant for No Event
    parser.add_argument('--t-no-event-exist', type=int, default=100,help="Time Tolerant for No Event Existence in seconds")


    return parser.parse_args()


def update_plot(frame, dict_parser:dict, args, dict_hist_patches, ax):
    global error_count,time_tolerant_no_event,bins_charge
    dict_v_charge_sum = {}
    for channel in dict_parser.keys():
        dict_v_charge_sum[channel] = []

    start_time = time.time()
    while time.time() - start_time < 1: # Calculate for each second
        try:
            # hacky way to reset the parser's event count
            for channel in dict_parser.keys():
                dict_parser[channel].n_entries = dict_parser[channel]._get_entries()
                event = dict_parser[channel].read_next()
                if event.id % 100 == 0:
                    print(f"Event {event.id}")
                event.record = event.record * -1
                event.record = event.record - np.mean(event.record[:100]) # Subtract Baseline
                dict_v_charge_sum[channel].append(np.sum(event.record[:-40]))
                error_count = 0

        except IndexError:
            error_count += 1
            print("At end of file. Error counts:\t", error_count)
            if error_count > time_tolerant_no_event:
                print(f"No new events for {time_tolerant_no_event} seconds, exiting")
                sys.exit()
            time.sleep(1)
    print("Setting datapoints")

    # Update Charge Spectrum
    global dict_bin_counts
    for channel in dict_hist_patches.keys():
        h = np.histogram(dict_v_charge_sum[channel], bins=bins_charge)
        dict_bin_counts[channel] += h[0]
        dict_hist_patches[channel].set_ydata(dict_bin_counts[channel])
    ax.set_ylim(0, np.max(np.concatenate(list(dict_bin_counts.values())))*1.1)
    ax.set_title(f"N of Triggers:{np.sum(list(dict_bin_counts.values())[0])}")
    return dict_hist_patches


def main():
    global dict_bin_counts, dict_hist_patches, time_tolerant_no_event, bins_charge
    args = parse_args()
    time_tolerant_no_event = args.t_no_event_exist
    # Check whether *.dat exist, if not, wait the WaveDump to write the new files
    while len(glob(args.input_directory + '/*.dat')) == 0:
        print("Waiting For WaveDump Writing Files...")
        time.sleep(2)

    for file in glob(args.input_directory + '/*.dat'):
        channel = file[-5]
        # Set Reader of Each Channels
        dict_parser[channel] = gdw.Parser(file, gdw.DigitizerFamily[args.digitizer])
        # Initialize Charge Histogram Counts
        dict_bin_counts[channel] = np.zeros(args.nbins-1)

    # binning Strategy
    bins_charge = np.linspace(args.bin_start, args.bin_end, args.nbins)
    bin_centers = 0.5 * (bins_charge[:-1] + bins_charge[1:])

    # Create initial plot
    fig, ax = plt.subplots()
    # Initial histogram setup
    for channel in dict_bin_counts.keys():
        (hist_patches, ) = ax.step(bin_centers, dict_bin_counts[channel], where='mid', label=f"ch{channel}")
        dict_hist_patches[channel] = hist_patches
    ax.set_xlabel("Total Charge [$ADC\cdot ns$]")
    ax.set_ylabel("Counts")
    ax.grid(True)
    ax.legend()
    # ax.set_title(
    #     f"Trigger sample: {args.trigger}, Short window: {args.shortWindow}, Long window: {args.longWindow}, Lookback: {args.lookback}, ADC threshold: {args.adcThreshold}")
    animation = FuncAnimation(
        fig, update_plot, fargs=(dict_parser, args, dict_hist_patches, ax), frames=None)
    plt.show()


if __name__ == "__main__":
    main()
