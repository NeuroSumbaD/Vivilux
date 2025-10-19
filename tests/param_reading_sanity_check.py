'''In this script we turn on one SFP Laser, and modulate a single parameter
    of the MZI over a range from 0 to 5 V while reading all output detectors.
    This is a simple sanity check that should reveal the periodicity of the MZI.    
'''

import __main__
from time import sleep

from vivilux.logger import log

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import vivilux.hardware.daq as daq
import vivilux.hardware.ser as ser
import vivilux.hardware.nidaq as ni

# TODO: Update board configuration to include NI, Ser and SFP boards

# ------------ Set up boards and netlist ------------
ni_boards =[
    ni.USB_6210("NI", 33703915, # NOTE: ni board is currently the only one with the necessary range for single-ended readout
        ni.AIPIN("PD_1_0", 0), 
        ni.AIPIN("PD_1_5", 5), 
        ni.AIPIN("PD_2_0", 6), 
        ni.AIPIN("PD_2_5", 11),
        ni.AIPIN("PD_3_0", 12),
        
        ni.AIPIN("PD_4_0", 1),
        ni.AIPIN("PD_5_0", 2),
        ni.AIPIN("PD_6_0", 3),
        
        ni.AIPIN("PD_3_5", 4),
        ni.AIPIN("PD_4_5", 7),
        ni.AIPIN("PD_5_5", 8),
        ni.AIPIN("PD_6_5", 9),
        
        ni.AIPIN("PD_1_2", 10),
        ni.AIPIN("PD_6_1", 13),
        ni.AIPIN("PD_1_3", 14),
        ni.AIPIN("PD_6_3", 15),
    ),
]

ser_boards = [
    ser.EVAL_AD5370("DAC", 0, 17, # 40-channel DAC with buffer board with csPin=17
        ser.AOPIN("3_5_o", 1), # NEW
        ser.AOPIN("3_5_i", 3),
        ser.AOPIN("1_5_i", 5),
        ser.AOPIN("2_4_o", 7), # NEW
        ser.AOPIN("2_4_i", 9),
        ser.AOPIN("3_3_o", 11), # NEW
        ser.AOPIN("3_3_i", 13),
        ser.AOPIN("1_3_o", 15),
        ser.AOPIN("1_3_i", 17),
        ser.AOPIN("2_2_o", 19), # NEW
        ser.AOPIN("2_2_i", 21),
        ser.AOPIN("3_1_i", 23),
        ser.AOPIN("1_1_i", 25),

        ser.AOPIN("5_1_i", 0),
        ser.AOPIN("4_2_i", 2),
        ser.AOPIN("4_2_o", 4), # NEW (mislabeled as 5_2_o on board?)
        ser.AOPIN("5_3_i", 6),
        ser.AOPIN("4_4_i", 8),
        ser.AOPIN("4_4_o", 10), # NEW (mislabeled as 5_4_o on board?)
        ser.AOPIN("5_5_i", 12),
        
        
        # New channels available if needed (currently unused):
        ser.AOPIN("2_1_i", 27),
        ser.AOPIN("3_2_i", 29),
        ser.AOPIN("2_3_i", 31),
        ser.AOPIN("3_4_i", 33),
        ser.AOPIN("2_5_i", 35),
        
        ser.AOPIN("4_1_i", 14),
        ser.AOPIN("5_2_i", 16),
        ser.AOPIN("4_3_i", 18),
        ser.AOPIN("5_4_i", 20),
        ser.AOPIN("4_5_i", 22),
    ),
]

pico = ser.BoardManager("PICO-001", *ser_boards)

fpga = ser.VC_709("VC-709",
                  ser.DIOPIN("Laser_0",0),
                  ser.DIOPIN("Laser_1",1),
                  ser.DIOPIN("Laser_2",2),
                  ser.DIOPIN("Laser_3",3),
                  )

# Create a netlist with the NI and MCC boards
ser.config_detected_devices([pico, fpga], verbose=False)
ni.config_detected_devices(ni_boards, verbose=False)
netlist = daq.Netlist(*ni_boards, pico, fpga)

# ------------ Set up experiment parameters ------------
with netlist:
    # Turn on single SFP Laser
    netlist["Laser_1"].dout(True)
    sleep(10)  # Allow time for laser to stabilize

    # Sweep a single MZI parameter from 0 to 5 V
    num_points = 50
    param_pin = "3_3_i"  # Pin to sweep
    netlist[param_pin].vout(2.5)
    pd_pins = [f"PD_{i}_5" for i in range(1, 7)]  # Photodetector pins to read
    readings_mat = np.zeros((num_points, len(pd_pins)))
    plus_mat = np.zeros((num_points, len(pd_pins)))
    minus_mat = np.zeros((num_points, len(pd_pins)))
    prior_measurements = np.array([netlist[pd_pin].vin() for pd_pin in pd_pins])
    for i, delta_voltage in enumerate(tqdm(np.linspace(0.05, 2.5, num_points), desc=f"Sweeping MZI parameter {param_pin}")):
        # forward step
        netlist[param_pin].vout(2.5 + delta_voltage)
        sleep(100e-3)  # Allow time for system to stabilize
        forward_measurements = np.array([netlist[pd_pin].vin() for pd_pin in pd_pins])
        # backward step
        netlist[param_pin].vout(2.5 - delta_voltage)
        sleep(100e-3)  # Allow time for system to stabilize
        backward_measurements = np.array([netlist[pd_pin].vin() for pd_pin in pd_pins])
        # derivative estimation
        readings = (forward_measurements - backward_measurements) / (2*delta_voltage)
        readings_mat[i] = readings
        plus_mat[i] = forward_measurements
        minus_mat[i] = backward_measurements
    
# Plot results
fig, axs = plt.subplots(len(pd_pins), figsize=(10, 6))
for i, pd_pin in enumerate(pd_pins):
    axs[i].plot(np.linspace(0, 2.5, num_points), readings_mat[:, i])
    axs[i].set_title(f"Readings from {pd_pin}")
    axs[i].set_xlabel(" Delta Voltage (V)")
    axs[i].set_ylabel("dV/dV")
# plt.tight_layout()

fig, axs = plt.subplots(len(pd_pins), figsize=(10, 6))
for i, pd_pin in enumerate(pd_pins):
    axs[i].plot(np.linspace(0, 2.5, num_points), plus_mat[:, i], label="Plus")
    axs[i].plot(np.linspace(0, 2.5, num_points), minus_mat[:, i], label="Minus")
    axs[i].set_title(f"Plus/Minus Readings from {pd_pin}")
    axs[i].set_xlabel(" Delta Voltage (V)")
    axs[i].set_ylabel("V")
    axs[i].legend()
plt.show()