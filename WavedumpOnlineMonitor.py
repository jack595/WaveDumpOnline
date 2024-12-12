# -*- coding:utf-8 -*-
# @Time: 2024/4/2 10:53
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: WavedumpOnlineMonitor.py
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

n_to_drop_in_waveform = None
n_total_triggers = 0
text_objects ={}


dict_parser = {}
dict_bin_counts = {}
dict_hist_patches = {}
dict_n_blank_waveform = {}

error_count = 0

def reset(event):
    print("Reset The Dataset.....")
    global dict_bin_counts, dict_n_blank_waveform,n_total_triggers
    n_total_triggers = 0
    for channel in dict_bin_counts.keys():
        dict_bin_counts[channel] = 0
        dict_n_blank_waveform[channel] = 0



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", help="input directory contained *.dat")
    parser.add_argument('--digitizer', choices=[member.name for member in gdw.DigitizerFamily],
                        help='Select a digitizer family', required=True)

    # Binning Strategy for Charge Spectrum
    parser.add_argument('--bin-start', type=float, default=-1000,help="Start of the charge binning")
    parser.add_argument('--bin-end', type=float, default=10000,help="End of the charge binning")
    parser.add_argument('--nbins', type=int, default=400,help="Number of bins for the charge histogram")

    # Threshold for Occupancy Calculation
    ## Waveform whose maximum are under threshold are view as blank events
    parser.add_argument('--threshold-ADC', type=int, default=5,help="Threshold")

    # Threshold ratio about peak to valley for cross talk removal, under threshold will be view as cross-talk events
    parser.add_argument('--threshold-peak-valley-ratio', type=float, default=0.5,help="Threshold about Peak to Valley Ratio for PMT Cross Talk Removal")
    parser.add_argument('--threshold-charge', type=float, default=300,help="Threshold about Charge for PMT Cross Talk Removal")

    # Time Tolerant for No Event
    parser.add_argument('--t-no-event-exist', type=int, default=100,help="Time Tolerant for No Event Existence in seconds")


    return parser.parse_args()

#############################################################################
##################### Occupancy Calculation Tools ################################
# Check is there a hit in the event, Tag whether it is blank waveform
def is_blank(waveform, charge, threshold_amp,
             threshold_peak_valley_ratio, threshold_charge):
    max_waveform = np.max(waveform)
    min_waveform = np.min(waveform)
    return (max_waveform < threshold_amp) & ( (max_waveform+min_waveform)/max_waveform<threshold_peak_valley_ratio) & (charge<threshold_charge)

def ErrorPropagationDivision(x,y, x_err, y_err):
    x = np.array(x)
    y = np.array(y)
    x_err = np.array(x_err)
    y_err = np.array(y_err)
    return ( (x_err/x)**2+(y_err/y)**2 )**0.5 * (x/y)

# def GetOccupancyFromBlank(n_Blank, nEvts_total):
#     Occupancy = 1-(n_Blank/nEvts_total)
#     Occupancy_Sigma = ErrorPropagationDivision(n_Blank, nEvts_total,
#                                                n_Blank**0.5, nEvts_total**0.5 )
#     return Occupancy, Occupancy_Sigma

def GetMuFromBlank(n_Blank, nEvts_total):
    mu = -np.log(n_Blank/nEvts_total)
    sigma = np.sqrt((1-np.exp(-mu) )/(nEvts_total*np.exp(-mu)))
    return mu, sigma
################################################################################



def update_plot(frame, dict_parser:dict, args, dict_hist_patches, ax):
    global error_count,time_tolerant_no_event,bins_charge, n_to_drop_in_waveform, n_total_triggers, text_objects
    dict_v_charge_sum = {}
    for channel in dict_parser.keys():
        dict_v_charge_sum[channel] = []

    start_time = time.time()
    while time.time() - start_time < 1: # Calculate for each second
        try:
            # hacky way to reset the parser's event count
            for i, channel in enumerate(dict_parser.keys()):
                dict_parser[channel].n_entries = dict_parser[channel]._get_entries()
                event = dict_parser[channel].read_next()
                event.record = np.array(event.record)*-1
                event.record = event.record - np.mean(event.record[:100]) # Subtract Baseline
                waveform = event.record[:n_to_drop_in_waveform]
                charge = np.sum(waveform)
                dict_n_blank_waveform[channel] += is_blank(waveform,charge, args.threshold_ADC,
                                                           args.threshold_peak_valley_ratio, args.threshold_charge)
                dict_v_charge_sum[channel].append(charge)
                error_count = 0
                if (i==0)&(event.id % 500 == 0) :
                    print(f"Event {event.id}")
            n_total_triggers += 1

        except IndexError:
            error_count += 1
            print("At end of file. Error counts:\t", error_count)
            if error_count > time_tolerant_no_event:
                print(f"No new events for {time_tolerant_no_event} seconds, exiting")
                sys.exit()
            time.sleep(1)

    global dict_bin_counts

    print("======> Setting datapoints")
    # Calculate Total Number of Triggers

    # Update Charge Spectrum
    for channel in dict_hist_patches.keys():
        h = np.histogram(dict_v_charge_sum[channel], bins=bins_charge)
        dict_bin_counts[channel] += h[0]
        dict_hist_patches[channel].set_ydata(dict_bin_counts[channel])
    ax.set_ylim(0, np.max(np.concatenate(list(dict_bin_counts.values())))*1.1)
    ax.set_title(f"N of Triggers:{n_total_triggers:.0f}")

    # Display Occupancy
    for channel, n_blank in dict_n_blank_waveform.items():
        mu, mu_err = GetMuFromBlank(n_blank, n_total_triggers)
        text_objects[channel].set_text( f"ch{channel}: {mu:.4f}+-{mu_err:.4f}, n_Blank: {n_blank}\n")

    return dict_hist_patches


def main():
    global dict_bin_counts, dict_hist_patches, time_tolerant_no_event, bins_charge, n_to_drop_in_waveform
    args = parse_args()

    if args.digitizer=="X742":
        n_to_drop_in_waveform = -40

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
        dict_n_blank_waveform[channel] = 0


    # binning Strategy
    bins_charge = np.linspace(args.bin_start, args.bin_end, args.nbins)
    bin_centers = 0.5 * (bins_charge[:-1] + bins_charge[1:])

    # Create initial plot
    fig, ax = plt.subplots()
    # Initial histogram setup
    for channel in dict_bin_counts.keys():
        (hist_patches, ) = ax.step(bin_centers, dict_bin_counts[channel], where='mid', label=f"ch{channel}")
        dict_hist_patches[channel] = hist_patches

    # initialize the occupancy text objects with some initial text
    global text_objects
    text_y = 0.5
    text_objects["title"] =  ax.text(0.95, text_y + 0.1, 'Occupancy Monitor', verticalalignment='bottom',
                                     horizontalalignment='right', transform=ax.transAxes)
    text_objects = {
    channel: ax.text(0.95, text_y - int(channel) * 0.05, '', verticalalignment='bottom',
                      horizontalalignment='right', transform=ax.transAxes)
    for channel in dict_n_blank_waveform.keys()
}

    # Place the button on the plot
    from matplotlib.widgets import Button
    ax_button = plt.axes([0.8, 0.9, 0.1, 0.075])  # Adjust these values as needed ([left, bottom, width, height])
    btn_reset = Button(ax_button, 'Reset')

    # Link the button event to the action
    btn_reset.on_clicked(reset)

    ax.set_xlabel("Total Charge [$ADC\cdot ns$]")
    ax.set_ylabel("Counts")
    ax.grid(True)
    ax.legend()
    animation = FuncAnimation(
        fig, update_plot, fargs=(dict_parser, args, dict_hist_patches, ax), frames=None)
    plt.show()


if __name__ == "__main__":
    main()
