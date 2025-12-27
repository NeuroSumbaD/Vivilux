'''Quick example of sweeping the top and bottom rows of MZIs to maximize the
    through state of the first row.
'''

from sfp_board_config_6x6 import netlist, fpga

import __main__
import os
from time import time, sleep
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.arbitrary_mzi import HardMZI_v3, gen_from_one_hot, gen_from_sparse_permutation
from vivilux.hardware.lasers import SFPLaserArray, SFPDetectorArray

voltage_sweep = np.linspace(0, 5.0, 50)  # Sweep from 0V to 5V

through_states = {}

# Define experiment within netlist context
with netlist:
    # Define the detector arrays before and after the MZI
    inputDetectors = DetectorArray(
        size=4,
        nets=["PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0",],# "PD_5_0", "PD_6_0",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms (TODO: double-check if these detectors are 220k or 10k)
    )
    raw_outputDetectors = DetectorArray(
        size=6,
        nets=["PD_1_5","PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5", "PD_6_5",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms
        min_zero=False, # Allow negative readings (negative possible, but unlikely, when using SFP lasers with vibrations instead of full on/off)
    )
    
    pd_nets = ["PD_1_2", "PD_1_3", "PD_1_5","PD_2_5",]
    off_readings = netlist.group_vin(pd_nets)
    pd_offsets = {net: off for net, off in zip(pd_nets, off_readings)}
    # print("PD offsets:", pd_offsets)
    
    pd_nets2 = ["PD_6_1", "PD_6_3", "PD_6_5","PD_5_5",]
    off_readings2 = netlist.group_vin(pd_nets2)
    pd_offsets.update({net: off for net, off in zip(pd_nets2, off_readings2)})
    print("PD offsets:", pd_offsets)

    # Define the laser array for the MZI input
    inputLaser = SFPLaserArray(
        size=4,
        control_nets=["laser_0", "laser_1", "laser_2", "laser_3"],
        detectors=inputDetectors,
        netlist=netlist,
        board=fpga,
        # use_vibrations=False,
        # pause=50e-3,
    )

    outputDetectors = SFPDetectorArray(
        detectors=raw_outputDetectors,
        lasers=inputLaser,  # Use the input laser for calibration
    )
    
    print("Turning laser 1 on (vibration).")
    inputLaser.setNormalized([1, 0, 0, 1])  # Set initial laser powers to 0
    
    print("Waiting 1 second for stabilization...")
    sleep(1)

    print("Sweeping PS_1_1_i to minimize PD_1_2...")
    first_sweep_current = np.zeros((len(pd_nets), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PS 1_1_i sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["1_1_i"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets)]
        first_sweep_current[:, i] = currents
        
    # Find minimum of PD_1_2
    max_idx_1_2 = np.argmin(first_sweep_current[0, :])
    optimal_voltage_1_1 = voltage_sweep[max_idx_1_2]
    through_states["1_1_i"] = optimal_voltage_1_1
    netlist["1_1_i"].vout(optimal_voltage_1_1)
    print(f"Optimal voltage for PS_1_1_i: {optimal_voltage_1_1:.3f} V")
    sleep(1)  # wait for stabilization
        
    print("Sweeping PS_1_3_i to minimize PD_1_3...")
    second_sweep_current = np.zeros((len(pd_nets), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PS 1_3_i sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["1_3_i"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets)]
        second_sweep_current[:, i] = currents
        
    # Find minimum of PD_1_3
    max_idx_1_3 = np.argmin(second_sweep_current[1, :])
    optimal_voltage_1_3 = voltage_sweep[max_idx_1_3]
    through_states["1_3_i"] = optimal_voltage_1_3
    netlist["1_3_i"].vout(optimal_voltage_1_3)
    print(f"Optimal voltage for PS_1_3_i: {optimal_voltage_1_3:.3f} V")
    sleep(1)  # wait for stabilization
    
    print("Sweeping PS_1_5_i to minimize PD_1_5...")
    third_sweep_current = np.zeros((len(pd_nets), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PS 1_5_i sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["1_5_i"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets)]
        third_sweep_current[:, i] = currents
        
    # Find minimum of PD_1_5
    max_idx_1_5 = np.argmin(third_sweep_current[2, :])
    optimal_voltage_1_5 = voltage_sweep[max_idx_1_5]
    through_states["1_5_i"] = optimal_voltage_1_5    
    netlist["1_5_i"].vout(optimal_voltage_1_5)
    print(f"Optimal voltage for PS_1_5_i: {optimal_voltage_1_5:.3f} V")
    sleep(1)  # wait for stabilization
    
    pd_nets2 = ["PD_6_1", "PD_6_3", "PD_6_5","PD_5_5",]
    print("Sweeping PS_6_1_i to minimize PD_6_1...")
    fourth_sweep_current = np.zeros((len(pd_nets2), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PS 6_1_i sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["6_1_i"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets2)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets2)]
        fourth_sweep_current[:, i] = currents
        
    # Find minimum of PD_6_1
    max_idx_6_1 = np.argmin(fourth_sweep_current[0, :])
    through_states["6_1_i"] = voltage_sweep[max_idx_6_1]
    optimal_voltage_6_1 = voltage_sweep[max_idx_6_1]
    netlist["6_1_i"].vout(optimal_voltage_6_1)
    print(f"Optimal voltage for PS_6_1_i: {optimal_voltage_6_1:.3f} V")
    sleep(1)  # wait for stabilization
    
    print("Sweeping PS_6_3_i to minimize PD_6_3...")
    fifth_sweep_current = np.zeros((len(pd_nets2), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PS 6_3_i sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["6_3_i"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets2)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets2)]
        fifth_sweep_current[:, i] = currents
        
    # Find minimum of PD_6_3
    max_idx_6_3 = np.argmin(fifth_sweep_current[1, :])
    optimal_voltage_6_3 = voltage_sweep[max_idx_6_3]
    through_states["6_3_i"] = optimal_voltage_6_3
    netlist["6_3_i"].vout(optimal_voltage_6_3)
    print(f"Optimal voltage for PS_6_3_i: {optimal_voltage_6_3:.3f} V")
    sleep(1)  # wait for stabilization
    
    print("Sweeping PS_6_5_i to minimize PD_6_5...")
    sixth_sweep_current = np.zeros((len(pd_nets2), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PS 6_5_i sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["6_5_i"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets2)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets2)]
        sixth_sweep_current[:, i] = currents
        
    # Find minimum of PD_6_5
    max_idx_6_5 = np.argmin(sixth_sweep_current[2, :])
    optimal_voltage_6_5 = voltage_sweep[max_idx_6_5]
    through_states["6_5_i"] = optimal_voltage_6_5
    netlist["6_5_i"].vout(optimal_voltage_6_5)
    print(f"Optimal voltage for PS_6_5_i: {optimal_voltage_6_5:.3f} V")
    sleep(1)  # wait for stabilization
    
    print("Checking effect of PHI phase shifters...")
    phi_nets = ["1_3_o", "1_5_o"]
    phi_sweep_current1 = np.zeros((len(pd_nets), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PHI 1_3_o sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["1_3_o"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets)]
        phi_sweep_current1[:, i] = currents
    netlist["1_3_o"].vout(0.0)  # reset PHI phase shifter
    sleep(1)
    
    phi_sweep_current2 = np.zeros((len(pd_nets), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PHI 1_5_o sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["1_5_o"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets)]
        phi_sweep_current2[:, i] = currents
        
    phi_nets2 = ["5_3_o", "5_5_o"]
    phi_sweep_current3 = np.zeros((len(pd_nets2), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PHI 5_3_o sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["5_3_o"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets2)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets2)]
        phi_sweep_current3[:, i] = currents
    netlist["5_3_o"].vout(0.0)  # reset PHI phase shifter
    sleep(1)
    phi_sweep_current4 = np.zeros((len(pd_nets2), len(voltage_sweep)))
    for i, voltage in enumerate(tqdm(voltage_sweep,
                        desc="PHI 5_5_o sweep",
                        unit="step",
                        total=len(voltage_sweep))):
        netlist["5_5_o"].vout(voltage)
        sleep(0.05)  # wait for settling
        readings = netlist.group_vin(pd_nets2)
        currents = [(pd_offsets[net]-reading)/220e3 for reading, net in zip(readings, pd_nets2)]
        phi_sweep_current4[:, i] = currents
    netlist["5_5_o"].vout(0.0)  # reset PHI phase shifter
    sleep(1)  # wait for stabilization
    
print("Saving through states to json...")
with open("through_states.json", "w") as f:
    json.dump(through_states, f, indent=4)
    
print("Plotting results...")
fig, axs = plt.subplots(1, 3, figsize=(10, 12))
axs[0].plot(voltage_sweep, first_sweep_current.T)
axs[0].set_title("Sweep of PS_1_1_i")
axs[0].set_xlabel("Voltage (V)")
axs[0].set_ylabel("Current (A)")
axs[0].legend(pd_nets)

axs[1].plot(voltage_sweep, second_sweep_current.T)
axs[1].set_title("Sweep of PS_1_3_i")
axs[1].set_xlabel("Voltage (V)")
axs[1].set_ylabel("Current (A)")
axs[1].legend(pd_nets)

axs[2].plot(voltage_sweep, third_sweep_current.T)
axs[2].set_title("Sweep of PS_1_5_i")
axs[2].set_xlabel("Voltage (V)")
axs[2].set_ylabel("Current (A)")
axs[2].legend(pd_nets)

fig, axs = plt.subplots(1, 3, figsize=(10, 12))
axs[0].plot(voltage_sweep, fourth_sweep_current.T)
axs[0].set_title("Sweep of PS_6_1_i")
axs[0].set_xlabel("Voltage (V)")
axs[0].set_ylabel("Current (A)")
axs[0].legend(pd_nets2)

axs[1].plot(voltage_sweep, fifth_sweep_current.T)
axs[1].set_title("Sweep of PS_6_3_i")
axs[1].set_xlabel("Voltage (V)")
axs[1].set_ylabel("Current (A)")
axs[1].legend(pd_nets2)
axs[2].plot(voltage_sweep, sixth_sweep_current.T)
axs[2].set_title("Sweep of PS_6_5_i")
axs[2].set_xlabel("Voltage (V)")
axs[2].set_ylabel("Current (A)")
axs[2].legend(pd_nets2)

# Plot PHI sweeps
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].plot(voltage_sweep, phi_sweep_current1.T)
axs[0].set_title("Sweep of PHI 1_3_o")
axs[0].set_xlabel("Voltage (V)")
axs[0].set_ylabel("Current (A)")
axs[0].legend(pd_nets)
axs[1].plot(voltage_sweep, phi_sweep_current2.T)
axs[1].set_title("Sweep of PHI 1_5_o")
axs[1].set_xlabel("Voltage (V)")
axs[1].set_ylabel("Current (A)")
axs[1].legend(pd_nets)

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].plot(voltage_sweep, phi_sweep_current3.T)
axs[0].set_title("Sweep of PHI 5_3_o")
axs[0].set_xlabel("Voltage (V)")
axs[0].set_ylabel("Current (A)")
axs[0].legend(pd_nets2)
axs[1].plot(voltage_sweep, phi_sweep_current4.T)
axs[1].set_title("Sweep of PHI 5_5_o")
axs[1].set_xlabel("Voltage (V)")
axs[1].set_ylabel("Current (A)")
axs[1].legend(pd_nets2)

plt.show()