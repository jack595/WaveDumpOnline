# WaveDump Live Monitor
Method learned from robjfoster's git@github.com:robjfoster/wavedumplivepsd.git and using WaveDump reader in robjfoster's repository

In time profile measurement, to monitor the charge distribution and occupancy of the waveform data from the digitizer in real time.
Also, position detector online monitor is implemented to monitor the position detector data in real time.

## Environment Settings
### Terminal
It is fine whether Powershell or Linux terminal to run the monitor.The only required package is `numpy`, `matplotlib` and `pandas` for the monitor.
Therefore, it is recommended to install the package by `pip install numpy matplotlib pandas` for the first run.

### WaveDump Settings (WaveDumpConfig.txt)
It is needed to mention that the data saved in Wavedump should be binary format rather than ASCII format, i.e. `OUTPUT_FILE_FORMAT  BINARY` . The binary format is more efficient and faster to read the data. In addition, 
event header should be turned on, corresponding to command line `OUTPUT_FILE_HEADER  YES` in WavedumpConfig.txt. 

Otherwise, the online monitor will not be able to read the data correctly and cause some errors.

## Usage 
0. Clone this repository:`git clone https://github.com/jack595/WaveDumpOnline.git`
1. Clone reader for Wavedump under WaveDumpOnline directory : `cd WaveDumpOnline && git clone https://github.com/robjfoster/gimmedatwave.git` 
2. Run the monitor:
* For charge spectrum monitor: `python WavedumpChargeOnlineMonitor.py --input-directory /mnt/f/Data_dEdxExperiment/Data_N6742/405nmLaser_test_pulse_width/ --digitizer X742`

* For position detector monitor which is needed to specified strip separation rule: `python WavedumpPositionOnlineMonitor.py --input-directory /mnt/f/Data_dEdxExperiment/Data_N6742/405nmLaser_test_pulse_width/ --digitizer X742 --separation-rule-file ./PostionReconstruction/StripSeparationRule.npz`

* For example, `job.sh` and `job_PositionReconstruction.sh` are the base script for running the monitor.
## Parameters
* `--input-directory` is the directory where the data is being written to by the digitizer, monitor will look for *.dat in this directory
* `--digitizer` is the digitizer being used, here only list X742 as example. Input can be X742, X740, X730, X725, X751

**Parameters for Occupancy**
* `--threshold-ADC` is the threshold of maximum of waveform
* `--threshold-peak-valley-ratio` is the threshold of peak-valley ratio:

$$ max+min \over max $$

* `--threshold-charge` is the threshold of the sum of the waveform.
So blank event definition: (max < threshold_ADC) or ( (max+min)/min < threshold_peak_valley_ratio) or (sum < threshold_charge)

**Optional Switcher**
* `--bin-start, --bin-end, --nbins` is the binning strategy. Configure the online charge binning for plot: `bin=np.linspace(bin_start, bin_end, nbins)`

## Occupancy Calculation

$$ \mu=-log{n_{Blank} \over n_{Total} }$$

$$ \sigma=\sqrt{ (1-exp{(-\mu)}) \over (n_{Total}*exp{(-\mu)}) } $$


## GUI Interface
The interface of charge spectrum monitor is shown as below:

![](./figure/GUI_Interface.png)

The interface of position detector monitor is shown as below:

![position_monitor.png](figure%2Fposition_monitor.png)