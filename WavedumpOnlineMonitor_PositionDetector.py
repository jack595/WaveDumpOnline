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
import pandas as pd

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
    global dict_bin_counts, dict_n_blank_waveform,n_total_triggers, h2d_position, bins_position
    n_total_triggers = 0
    for channel in dict_bin_counts.keys():
        dict_bin_counts[channel] = 0
        dict_n_blank_waveform[channel] = 0
    h2d_position = np.zeros((len(bins_position)-1, len(bins_position)-1))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", help="input directory contained *.dat")
    parser.add_argument('--digitizer',default="X742", choices=[member.name for member in gdw.DigitizerFamily],
                        help='Select a digitizer family')

    ################ Charge Spectrum Setting ################
    # Binning Strategy for Charge Spectrum
    parser.add_argument('--bin-start', type=float, default=-1000,help="Start of the charge binning")
    parser.add_argument('--bin-end', type=float, default=10000,help="End of the charge binning")
    parser.add_argument('--nbins', type=int, default=400,help="Number of bins for the charge histogram")
    # Set Baseline Range
    parser.add_argument('--n-baseline', type=int, default=50,help="Number of beginning waveform to get baseline")
    # Integral Window Range
    parser.add_argument('--integral-start', type=int, default=50,help="Start of the integral window")
    parser.add_argument('--integral-end', type=int, default=250,help="End of the integral window")
    # Polarity Setting
    parser.add_argument('--polarity', type=str, default="positive",
                        help="Polarity of Signal, positive or negative")
    ########################################################

    ################ Position Reconstruction Setting ################
    parser.add_argument('--charge-threshold-XY-layer', type=float, default=0.,help="Charge Threshold for X and Y Layer")
    parser.add_argument('--separation-rule-file', type=str, default="separation_rule.npz") # Can be attained from\
    #C:\Users\WenLJ\PycharmProjects\sk_DSNB\junofs_500G\LSPositionDetectorTest\code\UnstandChannelDecodingWithProcessData.ipynb

    # Time Tolerant for No Event
    parser.add_argument('--t-no-event-exist', type=int, default=100,help="Time Tolerant for No Event Existence in seconds")

    return parser.parse_args()


def LoadSeparationRule(file):
    with np.load(file,
                 allow_pickle=True) as f:
        dict_strip_range = f["dict_range"].item()
        dict_channel_pair = f["dict_channel_pair"].item()
        range_integral = f["range_integral"]
        dict_bins = f["dict_bins"].item()
    print("Integral Range:\t", range_integral)
    return range_integral, dict_strip_range, dict_channel_pair, dict_bins

# Position Reconstruction Tools (Copied from \
# C:\Users\WenLJ\PycharmProjects\sk_DSNB\junofs_500G\LSPositionDetectorTest\code\PositionReconstructionTools.py)
def SetLayerVariables(df_wave_filter:pd.DataFrame, dict_channel_pair=None):
    if dict_channel_pair is None:
        dict_channel_pair = {"X1":(0, 1), "X2":(2,3),"Y1":(4,5), "Y2":(6,7)}
    for key, value in dict_channel_pair.items():
        df_wave_filter["Q_Ratio_"+key] = (df_wave_filter[f"Q_ch{value[1]}"])/(df_wave_filter[f"Q_ch{value[1]}"]+df_wave_filter[f"Q_ch{value[0]}"])
        df_wave_filter["Q_Subtraction_Ratio_"+key] = (df_wave_filter[f"Q_ch{value[0]}"]-df_wave_filter[f"Q_ch{value[1]}"])/(df_wave_filter[f"Q_ch{value[1]}"]+df_wave_filter[f"Q_ch{value[0]}"])
        df_wave_filter["Q_sum_layer_"+key] = (df_wave_filter[f"Q_ch{value[1]}"]+df_wave_filter[f"Q_ch{value[0]}"])

def PositionRecForDf(df_wave_filter,dict_bins, dx_strip = 14.5, full_Q=False,
                     key_prefix_charge="Q_sum_layer_", suffix_output=""):
    """

    :param df_wave_filter: dataframe for LS strip detector waveform dataset
    :param dict_bins: dict of bins for each strip
    :param dx_strip: width of each strip
    :return:
    """
    suffix_key = "_full" if full_Q else ""
    for name_strip, bins_strip in dict_bins.items():
        df_wave_filter[f"Number_{name_strip}"] = pd.cut(df_wave_filter[f"Q_Ratio_{name_strip}"],
                                                             bins=bins_strip, include_lowest=True, right=True, labels=False)
        if name_strip=="X1":
            position_offset = 0
        elif name_strip=="X2":
            position_offset = dx_strip/2
        elif name_strip=="Y2":
            position_offset = 0
        elif name_strip=="Y1":
            position_offset = dx_strip/2
        else:
            position_offset = np.nan
            print(name_strip,"not end with 1 or 2, exit....")

        df_wave_filter[f"Position_{name_strip}"] = df_wave_filter["Number_"+name_strip]*dx_strip+position_offset
    for position in ["X", "Y"]:
        df_wave_filter[f"Position_{position}{suffix_output}"] = (df_wave_filter[f"Position_{position}1"]*df_wave_filter[f"{key_prefix_charge}{position}1{suffix_key}"]+df_wave_filter[f"Position_{position}2"]*df_wave_filter[f"{key_prefix_charge}{position}2{suffix_key}"])\
                                             /(df_wave_filter[f"{key_prefix_charge}{position}1{suffix_key}"]+df_wave_filter[f"{key_prefix_charge}{position}2{suffix_key}"])
    return df_wave_filter



def update_plot(frame, dict_parser:dict, args, dict_hist_patches, axes, im_position):
    # System settings
    global error_count,time_tolerant_no_event, n_total_triggers
    # Charge calculation setting
    global polarity, n_baseline, integral_start, integral_end,bins_charge, n_to_drop_in_waveform
    # Position Reconstruction Setting
    global bins_position,h2d_position, dict_channel_pair, dict_bins,charge_threshold_for_XY_layer

    dict_v_charge_sum = {}
    for channel in dict_parser.keys():
        dict_v_charge_sum[f"Q_ch{channel}"] = []

    start_time = time.time()
    while time.time() - start_time < 5: # Calculate for each second
        try:
            # hacky way to reset the parser's event count
            for i, channel in enumerate(dict_parser.keys()):
                dict_parser[channel].n_entries = dict_parser[channel]._get_entries()
                event = dict_parser[channel].read_next()
                event.record = event.record*polarity
                event.record = event.record - np.mean(event.record[:n_baseline]) # Subtract Baseline
                waveform = event.record[:n_to_drop_in_waveform]
                charge = np.sum(waveform[integral_start:integral_end])
                dict_v_charge_sum[f'Q_ch{channel}'].append(charge)
                error_count = 0
            n_total_triggers += 1
            df_data = pd.DataFrame.from_dict(dict_v_charge_sum)
            SetLayerVariables(df_data, dict_channel_pair)
            index_charge_threshold_for_XY_layer = ((df_data["Q_sum_layer_X1"]+df_data["Q_sum_layer_X2"]>charge_threshold_for_XY_layer) &
                                                   (df_data["Q_sum_layer_Y1"]+df_data["Q_sum_layer_Y2"]>charge_threshold_for_XY_layer))
            df_data = df_data[index_charge_threshold_for_XY_layer]
            PositionRecForDf(df_data, dict_bins, dx_strip=14.5, full_Q=False,
                                       key_prefix_charge="Q_sum_layer_", suffix_output="")

            h2d_position_tmp,_, _ =  np.histogram2d(df_data["Position_X"],
                                               df_data["Position_Y"],
                                                bins=bins_position)
            h2d_position += h2d_position_tmp

        except IndexError:
            error_count += 1
            print("At end of file. Error counts:\t", error_count)
            if error_count > time_tolerant_no_event:
                print(f"No new events for {time_tolerant_no_event} seconds, exiting")
                sys.exit()
            time.sleep(1)

    global dict_bin_counts

    print("======> Setting datapoints")

    # Update Data for Plot
    im_position.set_data(h2d_position.T)
    im_position.set_clim(0, h2d_position.max()*1.1)

    for channel in dict_hist_patches.keys():
        h = np.histogram(dict_v_charge_sum[f'Q_ch{channel}'], bins=bins_charge)
        dict_bin_counts[channel] += h[0]
        dict_hist_patches[channel].set_ydata(dict_bin_counts[channel])

    axes[0].set_ylim(0, np.max(np.concatenate(list(dict_bin_counts.values())))*1.1)
    axes[1].set_title(f"N of Triggers:{n_total_triggers:.0f}")

    return list(dict_hist_patches.values()) + [im_position]


def main():
    ################# Initialize Global Variables #################
    # Charge calculation setting
    global polarity, n_baseline, integral_start, integral_end,bins_charge, n_to_drop_in_waveform
    # System settings
    global error_count,time_tolerant_no_event, n_total_triggers
    # Position Reconstruction Setting
    global bins_position,h2d_position, dict_channel_pair, dict_bins,charge_threshold_for_XY_layer
    args = parse_args()

    # Set Polarity
    if args.polarity == "negative":
        polarity = -1
    elif args.polarity == "positive":
        polarity = 1
    else:
        raise ValueError("Polarity should be positive or negative!!!")

    if args.digitizer=="X742":
        n_to_drop_in_waveform = -40

    # Charge Setting
    n_baseline = args.n_baseline
    integral_end = args.integral_end
    integral_start = args.integral_start

    time_tolerant_no_event = args.t_no_event_exist

    # Load Separation Rule
    range_integral, dict_strip_range, dict_channel_pair, dict_bins = LoadSeparationRule(args.separation_rule_file)
    charge_threshold_for_XY_layer = args.charge_threshold_XY_layer
    ################################################################

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

    bins_position = np.arange(-10, 180, 1)
    h2d_position = np.zeros((len(bins_position)-1, len(bins_position)-1))

    # binning Strategy
    bins_charge = np.linspace(args.bin_start, args.bin_end, args.nbins)
    bin_centers = 0.5 * (bins_charge[:-1] + bins_charge[1:])

    # Create initial plot
    fig, axes = plt.subplots(1, 2)

    # Initial histogram setup
    ## For position plot
    im_position = axes[1].imshow(h2d_position.T, origin='lower', aspect='auto', extent=[bins_position[0], bins_position[-1],
                                                                  bins_position[0], bins_position[-1]])
    ## For charge spectrum
    for channel in dict_bin_counts.keys():
        (hist_patches, ) = axes[0].step(bin_centers, dict_bin_counts[channel], where='mid', label=f"ch{channel}")
        dict_hist_patches[channel] = hist_patches

    # Place the button on the plot
    from matplotlib.widgets import Button
    ax_button = plt.axes([0.8, 0.9, 0.1, 0.075])  # Adjust these values as needed ([left, bottom, width, height])
    btn_reset = Button(ax_button, 'Reset')

    # Link the button event to the action
    btn_reset.on_clicked(reset)

    axes[0].set_xlabel("Total Charge [$ADC\cdot ns$]")
    axes[0].set_ylabel("Counts")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("X [mm]")
    axes[1].set_ylabel("Y [mm]")
    axes[1].set_title("Position Monitor")
    plt.colorbar(im_position, ax=axes[1])

    animation = FuncAnimation(
        fig, update_plot, fargs=(dict_parser, args, dict_hist_patches, axes,
                                 im_position), frames=100)


    plt.show()

if __name__ == "__main__":
    main()
