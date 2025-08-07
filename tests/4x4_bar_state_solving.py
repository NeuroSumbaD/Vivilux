'''This module attempts to solve for bar state control voltages to use
    to use a 4x4 subset of the 6x6 MZI board. To avoid coupling light 
    to rows of the matrix that are not being used, the MZIs to those
    rows must have 100% "through" coupling and no "cross" coupling.
    
    The bar state control voltages are saved to a JSON file with the
    key names as the net name and the value as the control voltage.
    The JSON file is saved in the same directory as this script.
    The file is named "4x4_bar_state_voltages.json".
'''

from time import sleep
import os
import __main__
import json
from typing import Optional

from vivilux.hardware.daq import Netlist
from board_config_6x6 import netlist
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import LaserArray
from vivilux.logger import log

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

num_points = 500
read_delay = 1000e-3
json_dict = {}
plt.ion()

def bar_state(detectors: DetectorArray,
              control_nets: list[str],
              laser_array: LaserArray,
              netlist: Netlist,
              limits: tuple[float, float] = (0, 5),
              num_points: int = 100,
              read_delay: float = 100e-3, # 100 ms default delay before reading the detectors
              laser_power: Optional[np.ndarray] = None,  # Optional, if None, use 0.5 V
              ) -> tuple[np.ndarray, np.ndarray]:
    '''Procedure for nulling the detectors using the bar state of the MZI.
    
        Args:
            detector_nets (list[str]): List of detector net names to read.
            control_nets (list[str]): List of control net names to set.
            laser_array (LaserArray): The laser array to use for the MZI input.
            netlist (Netlist): The netlist to use for the MZI.
            limits (tuple[float, float]): The control signal limits in volts.
            num_points (int): The number of points to sweep the control voltages.
            read_delay (float): Delay before reading the detectors in seconds.
        Returns:
            np.ndarray: The control voltages that null the detectors.
    '''
    print(f"Setting {control_nets} to bar state (reading from {detectors.nets}).")
    log.info(f"Setting {control_nets} to bar state (reading from {detectors.nets}).")

    nulled_voltage = np.zeros(len(control_nets))
    min_readings = np.full(len(detectors.nets), np.inf)

    history = np.zeros((num_points, len(detectors.nets)))

    # zero the control nets
    for net in control_nets:
        netlist[net].vout(0)
        
    # turn on all the lasers
    # laser_power = np.full(len(laser_array.control_nets), laser_array.limits[1])
    # laser_array.setControl(laser_power)
    if laser_power is None:
        laser_power = np.full(len(laser_array.control_nets), 0.5)
    laser_array.setNormalized(laser_power)
    sleep(5) # 5 SECOND DELAY AFTER LASERS TURN ON (for stabilization)
        
    # sweep control voltages over the range
    voltages = np.linspace(limits[0], limits[1], num_points)
    for volt_index, volt in tqdm(enumerate(voltages), total=num_points, desc=f"Sweeping voltages to minimize detectors: {detectors.nets}"):
        # print(f"Voltage {volt_index+1}/{num_points}: Setting control voltages to {volt:.2f} V.")
        log.info(f"Voltage {volt_index+1}/{num_points}: Setting control voltages to {volt:.2f} V.")
        # set the control voltages
        for net in control_nets:
            netlist[net].vout(volt)
        sleep(read_delay)
        
        # read the detector values
        # for index, net in enumerate(detector_nets):
        #     reading = netlist[net].vin()
        #     print(f"{net}: {reading} V")
        #     log.info(f"{net}: {reading} V")
        #     history[volt_index, index] = reading
            
        #     if reading < min_readings[index]:
        #         min_readings[index] = reading
        #         nulled_voltage[index] = volt
        #         print(f"FOUND NEW MINIMUM: {net} at {volt} V")
        #         log.info(f"FOUND NEW MINIMUM: {net} at {volt} V")
        readings = detectors.read() * 1e6  # Convert to uA
        history[volt_index, :] = readings
        for index, net in enumerate(detectors.nets):
            if readings[index] < min_readings[index]:
                min_readings[index] = readings[index]
                nulled_voltage[index] = volt
                # print(f"FOUND NEW MINIMUM: {net} at {volt:.2f} V")
                log.info(f"FOUND NEW MINIMUM: {net} at {volt:.2f} V")
        
            
    # turn off all lasers
    laser_array.setControl(np.zeros(len(laser_array.control_nets)))
    
    return nulled_voltage, history, voltages

def plot_history(history: np.ndarray,
                 voltages: np.ndarray,
                 detector_nets: list[str],
                 ):
    plt.figure()
    plt.plot(voltages, history)
    plt.xlabel("Control Voltage (V)")
    plt.ylabel("Detector reading (uA)")
    plt.title("Detector Reading vs MZI Voltage")
    plt.legend(detector_nets)
    plt.show()

with netlist:
    
    # Define the detector arrays before and after the MZI
    inputDetectors = DetectorArray(
        size=4,
        nets=["PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0",],# "PD_5_0", "PD_6_0",],
        netlist=netlist,
    )
    outputDetectors = DetectorArray(
        size=4,
        nets=["PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5",],# "PD_5_5", "PD_6_5",],
        netlist=netlist,
    )
    
    col1 = DetectorArray(
        size=2,
        nets=["PD_1_2", "PD_6_1"],
        netlist=netlist,
    )
    
    col3 = DetectorArray(
        size=2,
        nets=["PD_1_3", "PD_6_3"],
        netlist=netlist,
    )
    
    col5 = DetectorArray(
        size=2,
        nets=["PD_1_5", "PD_6_5"],
        netlist=netlist,
    )

    # Define the laser array for the MZI input
    inputLaser = LaserArray(
        size=4, # 6
        control_nets=["laser_1", "laser_2", "laser_3", "laser_4"],# "laser_5", "laser_6"],
        detectors=inputDetectors,  # Use the input detectors for calibration
        limits=(0, 10),  # Control signal limits in-10 volts
        netlist=netlist,
    )
    
    # Minimize PD_1_2 and PD_6_1 by tuning 1_1_i and 5_1_i
    voltages, history, xaxis = bar_state(
        detectors=col1,
        control_nets=["1_1_i", "5_1_i"],
        laser_array=inputLaser,
        netlist=netlist,
        num_points=num_points,
        read_delay=read_delay,  # 100 ms delay before reading the detectors
        laser_power=np.array([1.0, 0.0, 0.0, 1.0]),
    )
    plot_history(history, xaxis, ["PD_1_2", "PD_6_1"])
    netlist["1_1_i"].vout(voltages[0])
    json_dict["1_1_i"] = voltages[0]
    netlist["5_1_i"].vout(voltages[1])
    json_dict["5_1_i"] = voltages[1]
    
    # Minimize PD_1_3 and PD_6_3 by tuning 1_3_i and 5_3_i
    voltages, history, xaxis = bar_state(
        detectors=col3,
        control_nets=["1_3_i", "5_3_i"],
        laser_array=inputLaser,
        netlist=netlist,
        num_points=num_points,
        read_delay=read_delay,  # 100 ms delay before reading the detectors
        laser_power=np.array([1.0, 0.0, 0.0, 1.0]),
    )
    plot_history(history, xaxis, ["PD_1_3", "PD_6_3"])
    netlist["1_3_i"].vout(voltages[0])
    json_dict["1_3_i"] = voltages[0]
    netlist["5_3_i"].vout(voltages[1])
    json_dict["5_3_i"] = voltages[1]

    # Minimize PD_1_5 and PD_6_5 by tuning 1_5_i and 5_5_i
    voltages, history, xaxis = bar_state(
        detectors=col5,
        control_nets=["1_5_i", "5_5_i"],
        laser_array=inputLaser,
        netlist=netlist,
        num_points=num_points,
        read_delay=read_delay,  # 100 ms delay before reading the detectors
        laser_power=np.array([1.0, 0.0, 0.0, 1.0]),
    )
    plot_history(history, xaxis, ["PD_1_5", "PD_6_5"])
    netlist["1_5_i"].vout(voltages[0])
    json_dict["1_5_i"] = voltages[0]
    netlist["5_5_i"].vout(voltages[1])
    json_dict["5_5_i"] = voltages[1]

# Save to JSON file in the same directory as this script
json_path = os.path.join(os.path.dirname(os.path.abspath(__main__.__file__)), "4x4_bar_state_voltages.json")
json.dump(json_dict, open(json_path, "w"), indent=4)

input("JSON file saved, press enter to quit...")