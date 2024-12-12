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
from sympy.printing.pretty.pretty_symbology import line_width

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from gimmedatwave.gimmedatwave import gimmedatwave as gdw  # nopep8

# Set Global Variables
global bins_charge,bins_charge_ratio
global bin_centers
global dict_bin_counts, dict_bin_counts_charge_ratio
global error_counts
global dict_parser
global time_tolerant_no_event # seconds


n_to_drop_in_waveform = None
n_total_triggers = 0
text_objects ={}


dict_parser = {}
dict_bin_counts = {}
dict_bin_counts_charge_ratio = {}
dict_hist_patches = {}
dict_n_blank_waveform = {}

error_count = 0

def GetBinCenter(bins):
    return 0.5 * (bins[:-1] + bins[1:])

def reset(event):
    print("Reset The Dataset.....")
    global dict_bin_counts,dict_bin_counts_charge_ratio, dict_n_blank_waveform,n_total_triggers, h2d_position, bins_position
    n_total_triggers = 0
    for channel in dict_bin_counts.keys():
        dict_bin_counts[channel] = 0
        dict_n_blank_waveform[channel] = 0
    for channel in dict_bin_counts_charge_ratio.keys():
        dict_bin_counts_charge_ratio[channel] = 0
    h2d_position = np.zeros((len(bins_position)-1, len(bins_position)-1))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", help="input directory contained *.dat")
    parser.add_argument('--digitizer',default="X742", choices=[member.name for member in gdw.DigitizerFamily],
                        help='Select a digitizer family')

    ################ Charge Spectrum Setting ################
    # Binning Strategy for Charge Spectrum
    parser.add_argument('--bin-start', type=float, default=-1000,help="Start of the charge binning")
    parser.add_argument('--bin-end', type=float, default=30000,help="End of the charge binning")
    parser.add_argument('--nbins', type=int, default=100,help="Number of bins for the charge histogram")
    parser.add_argument('--nbins-charge-ratio', type=int, default=200,help="Number of bins for the charge ratio histogram")
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
    parser.add_argument('--charge-threshold-XY-layer', type=float, default=2000.,help="Charge Threshold for X and Y Layer")
    parser.add_argument('--charge-ratio-threshold-XY', type=float, default=2000.,help="Charge Threshold for X and Y Layer")
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
def SetLayerVariables(df_wave_filter, dict_channel_pair=None):
    if dict_channel_pair is None:
        dict_channel_pair = {"X1":(0, 1), "X2":(2,3),"Y1":(4,5), "Y2":(6,7)}
    for key, value in dict_channel_pair.items():
        df_wave_filter["Q_Ratio_"+key] = (df_wave_filter[f"Q_ch{value[1]}"])/(df_wave_filter[f"Q_ch{value[1]}"]+df_wave_filter[f"Q_ch{value[0]}"])
        # df_wave_filter["Q_Subtraction_Ratio_"+key] = (df_wave_filter[f"Q_ch{value[0]}"]-df_wave_filter[f"Q_ch{value[1]}"])/(df_wave_filter[f"Q_ch{value[1]}"]+df_wave_filter[f"Q_ch{value[0]}"])
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
    global dict_bin_counts, dict_bin_counts_charge_ratio
    # Position Reconstruction Setting
    global bins_position,h2d_position, dict_channel_pair, dict_bins,charge_threshold_for_XY_layer

    dict_Q = {}
    for channel in dict_parser.keys():
        dict_Q[f"Q_ch{channel}"] = np.array([])

    start_time = time.time()
    while time.time() - start_time < 5: # Calculate for each second
        try:
            # t_start_reading = time.time()
            # hacky way to reset the parser's event count
            for i, channel in enumerate(dict_parser.keys()):
                dict_parser[channel].n_entries = dict_parser[channel]._get_entries()
                event = dict_parser[channel].read_next()
                event.record = event.record*polarity
                event.record = event.record - np.mean(event.record[:n_baseline]) # Subtract Baseline
                waveform = event.record[:n_to_drop_in_waveform]
                charge = np.sum(waveform[integral_start:integral_end])
                dict_Q[f'Q_ch{channel}'] = np.append(dict_Q[f'Q_ch{channel}'], charge)
                error_count = 0
            # t_end_reading = time.time()

            for key in dict_Q.keys():
                dict_Q[key] = np.array(dict_Q[key])
            n_total_triggers += 1
            t_finish_convert_df = time.time()
            SetLayerVariables(dict_Q, dict_channel_pair)

            # t_set_variables = time.time()

            # Charge Spectrum
            for name_layer in dict_bin_counts.keys():
                if name_layer =="X+Y":
                    dict_bin_counts[name_layer] += np.histogram(dict_Q[f'Q_sum_layer_X1']+dict_Q[f'Q_sum_layer_X2']+\
                                                                dict_Q[f'Q_sum_layer_Y1']+dict_Q[f'Q_sum_layer_Y2'], bins=bins_charge)[0]
                else:
                    dict_bin_counts[name_layer] += np.histogram(dict_Q[f'Q_sum_layer_{name_layer}1']+dict_Q[f'Q_sum_layer_{name_layer}2'], bins=bins_charge)[0]

            # Charge Ratio
            for name_layer in dict_bin_counts_charge_ratio.keys():
                dict_bin_counts_charge_ratio[name_layer] += np.histogram(dict_Q[f'Q_Ratio_{name_layer}'][dict_Q[f"Q_sum_layer_{name_layer}"]>args.charge_ratio_threshold_XY],
                                                                         bins=bins_charge_ratio)[0]

            # Position Reconstruction
            PositionRecForDf(dict_Q, dict_bins, dx_strip=14.5, full_Q=False,
                                       key_prefix_charge="Q_sum_layer_", suffix_output="")

            index_charge_threshold_for_XY_layer = ((dict_Q["Q_sum_layer_X1"]+dict_Q["Q_sum_layer_X2"]>charge_threshold_for_XY_layer) &
                                                   (dict_Q["Q_sum_layer_Y1"]+dict_Q["Q_sum_layer_Y2"]>charge_threshold_for_XY_layer))
            for key in dict_Q.keys():
                dict_Q[key] = dict_Q[key][index_charge_threshold_for_XY_layer]
            # t_end_get_variables = time.time()

            h2d_position_tmp,_, _ =  np.histogram2d(dict_Q["Position_X"],
                                               dict_Q["Position_Y"],
                                                bins=bins_position)
            h2d_position += h2d_position_tmp
            # print(f"Reading Time:\t{t_end_reading-t_start_reading:.2g}\tConvert DF Time:\t{t_finish_convert_df-t_end_reading:.2g}\tSet Variables Time:\t{t_set_variables-t_finish_convert_df:.2g}\tGet Variables Time:\t{t_end_get_variables-t_set_variables:.2g}")

        except IndexError:
            error_count += 1
            print("At end of file. Error counts:\t", error_count)
            if error_count > time_tolerant_no_event:
                print(f"No new events for {time_tolerant_no_event} seconds, exiting")
                sys.exit()
            time.sleep(1)


    print("======> Setting datapoints")

    # Update Data for Plot
    im_position.set_data(h2d_position.T)
    im_position.set_clim(0, h2d_position.max()*1.1)

    for name_layer in dict_hist_patches.keys():
        if name_layer[-1]=="1" or name_layer[-1]=="2":
            dict_hist_patches[name_layer].set_ydata(dict_bin_counts_charge_ratio[name_layer])
        else:
            dict_hist_patches[name_layer].set_ydata(dict_bin_counts[name_layer])

    axes[0].set_ylim(0, np.max(np.concatenate(list(dict_bin_counts.values())))*1.1)
    axes[1].set_title(f"N of Triggers:{n_total_triggers:.0f}")
    for ax in axes[2]:
        ax.set_ylim(0, np.max(np.concatenate(list(dict_bin_counts_charge_ratio.values())))*1.1)

    return list(dict_hist_patches.values()) + [im_position]


def main():
    ################# Initialize Global Variables #################
    # Charge calculation setting
    global polarity, n_baseline, integral_start, integral_end,bins_charge,bins_charge_ratio, n_to_drop_in_waveform
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

    # Charge Spectrum Setting
    v_keys_charge_spectrum = ["X", "Y", "X+Y"]
    for key in v_keys_charge_spectrum:
        # Initialize Charge Histogram Counts
        dict_bin_counts[key] = np.zeros(args.nbins-1)

    # Charge Ratio Setting
    v_keys_charge_ratio = ["X1", "X2", "Y1", "Y2"]
    for key in v_keys_charge_ratio:
        dict_bin_counts_charge_ratio[key] = np.zeros(args.nbins_charge_ratio-1)

    bins_position = np.arange(-10, 180, 1)
    h2d_position = np.zeros((len(bins_position)-1, len(bins_position)-1))

    # binning Strategy
    bins_charge = np.linspace(args.bin_start, args.bin_end, args.nbins)
    bin_centers = GetBinCenter(bins_charge)
    bins_charge_ratio = np.linspace(0, 1, args.nbins_charge_ratio)
    bin_centers_charge_ratio = GetBinCenter(bins_charge_ratio)

    # Create initial plot
    # fig, axes = plt.subplots(1, 3)
    fig = plt.figure(figsize=(15, 5))

    # 左边两张大图
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])  # 定义三列的网格
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # 右边拆成四张竖直对齐的子图
    gs_right = gs[0, 2].subgridspec(4, 1)  # 在右边第三列再分为四行
    ax3_1 = fig.add_subplot(gs_right[0, 0])
    ax3_2 = fig.add_subplot(gs_right[1, 0])
    ax3_3 = fig.add_subplot(gs_right[2, 0])
    ax3_4 = fig.add_subplot(gs_right[3, 0])

    axes = [ax1,ax2, [ax3_1, ax3_2, ax3_3, ax3_4]]

    # Initial histogram setup
    ## For position plot
    im_position = axes[1].imshow(h2d_position.T, origin='lower', aspect='auto', extent=[bins_position[0], bins_position[-1],
                                                                  bins_position[0], bins_position[-1]])
    position_center = 80
    axes[1].axhline(position_center, color='r', linestyle='--',linewidth=1, alpha=0.5)
    axes[1].axvline(position_center, color='r', linestyle='--',linewidth=1, alpha=0.5)

    ## For charge spectrum
    axes[0].axvline(charge_threshold_for_XY_layer, color='r', linestyle='--', label="Charge Threshold\nfor XY Layer")
    dict_colors = {'X':"b", "Y":"g", "X+Y":"k"}
    for name_layer in dict_bin_counts.keys():
        (hist_patches, ) = axes[0].step(bin_centers, dict_bin_counts[name_layer], where='mid', label=f"{name_layer}",
                                        color=dict_colors[name_layer])
        dict_hist_patches[name_layer] = hist_patches

    # For Charge Ratio
    for i,key in enumerate(dict_bin_counts_charge_ratio.keys()):
        (hist_patches_charge_ratio, ) = axes[2][i].step(bin_centers_charge_ratio, dict_bin_counts_charge_ratio[key],
                                                        where='mid', label=f"{key}",color="r")
        dict_hist_patches[key] = hist_patches_charge_ratio
        axes[2][i].set_title(key)
        for value in dict_bins[key]:
            axes[2][i].axvline(value, color='r', linestyle='--', alpha=0.5,linewidth=0.5)

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

    axes[2][0].xaxis.set_visible(False)
    axes[2][1].xaxis.set_visible(False)
    axes[2][2].xaxis.set_visible(False)
    axes[2][-1].set_xlabel("$\\frac{Q2}{Q1+Q2}$")

    animation = FuncAnimation(
        fig, update_plot, fargs=(dict_parser, args, dict_hist_patches, axes,
                                 im_position), frames=100)

    # fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
