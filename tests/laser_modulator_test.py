'''In this test we are sweeping the voltage range of the modulator pins to
    examine the range of power swept by each laser input in isolation.
'''

from time import sleep

from sfp_board_config_6x6 import netlist, fpga

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

voltages = np.linspace(0, 10, 100)
pd_channels = [f"PD_{index}_0" for index in range(1,7)]
data = np.zeros((4, len(pd_channels), len(voltages))) # preallocated data array

with netlist:
    fpga.update_vibration([1, 1, 1, 1]) # Try without SFP oscillation?
    fpga.update_lasers([0, 0, 0, 0]) # Initialize lasers off
    
    pd_offsets = np.array([netlist[pin].vin() for pin in pd_channels])
    
    # turn on one laser at a time and sweep voltages
    for laser_index in range(4):
        netlist[f"Laser_{laser_index}"].dout(True) # Turn on
        print(f"Turning on laser {laser_index}. Waiting 30s for settling...")
        sleep(30)
        
        for volt_index, voltage in tqdm(enumerate(voltages),
                                        desc=f"Sweeping modulator for laser {laser_index}",
                                        total=len(voltages)):
            netlist[f"Mod_{laser_index}"].vout(voltage)
            sleep(1)
            for pin_index, pin_name in enumerate(pd_channels):
               data[laser_index, pin_index, volt_index] = netlist[pin_name].vin()
            # data[laser_index, :, volt_index] = netlist.group_vin(pd_channels)
            
        netlist[f"Mod_{laser_index}"].vout(voltage)
        
        netlist[f"Laser_{laser_index}"].dout(False) # Turn on
        
        
def show_plot(data: np.ndarray, title: str, ylabel: str):
    plt.figure()
    plt.plot(voltages, data)
    plt.title(title)
    plt.legend(pd_channels)
    plt.xlabel("voltage (V)")
    plt.ylabel(ylabel)
    plt.show(block=False)
        
def plot_raw(raw_data: np.ndarray, title: str):
    show_plot(raw_data.T, title="Raw data " + title, ylabel="raw data (V)")

def plot_current(raw_data: np.ndarray, offsets: np.ndarray, title: str):
    offsets = offsets if offsets is not None else np.zeros_like(raw_data)
    
    currents = (offsets[:, np.newaxis] - raw_data) / 220e3 # convert using transimpedance
    
    show_plot(currents.T, title="Photocurrents " + title, ylabel="current (A)")
    

# plot each photocurrent result
for laser_index in range(4):
    plot_current(data[laser_index], pd_offsets, f"for laser {laser_index}")
    
plt.show()

# plot each raw
for laser_index in range(4):
    plot_raw(data[laser_index], f"for laser {laser_index}")