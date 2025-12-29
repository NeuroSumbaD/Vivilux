'''This module is a configuration intended for calibrating the laser
    polarization controllers on the 6x6 MZI mesh on the AIM 2020 tapeout.
    In this module it is expected that the lasers are controlled through
    DAQami external to the script. See board_config_6x6.py for similar
    calibration with the lasers controlled through the script.
    
    Note: In this script, I detect both first and last columns of the
    detectors to see how much power is being lost between these columns.
'''

import vivilux.hardware.daq as daq
import vivilux.hardware.mcc as mcc
import vivilux.hardware.nidaq as ni

from board_config_6x6 import netlist

if __name__ == "__main__":
    from time import sleep
    from vivilux.hardware.visualization import HeatmapVisualizer
    from vivilux.hardware.detectors import DetectorArray
    import numpy as np
    np.set_printoptions(suppress=True,)    

    detector_nets = ["PD_1_0", "PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0", "PD_6_0",
                     "PD_1_4", "PD_2_4", "PD_3_4",  "PD_4_4", "PD_5_4", "PD_6_4",
                     ]
    detectors_state = HeatmapVisualizer(shape=(6, 2), vmin=0, vmax=1.5,
                                        xlabel="Detector Column",
                                        ylabel="Detector Row",
                                        clabel="Photocurrent (uA)")
    
    delay_read = 1000e-3 # ms delay before reading the detectors
    
    # Test streaming data from the NI board
    with netlist:
        # initialize the detector array
        detector_array = DetectorArray(12,
                                       nets=detector_nets,
                                       netlist=netlist,
                                       transimpedance=220e3,  # 220k ohms
                                       )
        while True:
            data = detector_array.read()
            data = data.reshape((2, 6)).T*1e6  # Reshape to 6x2 matrix and convert to uA
            print("Detector readings (uA):")
            print(data)
            print(f"First column total current: {data[:, 0].sum():.2f} uA")
            print(f"Last column total current: {data[:, 1].sum():.2f} uA")
            detectors_state.update(data)
            sleep(delay_read)