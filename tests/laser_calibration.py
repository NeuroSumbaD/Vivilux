'''This module is a configuration intended for calibrating the laser
    polarization controllers on the 6x6 MZI mesh on the AIM 2020 tapeout.
    In this module it is expected that the lasers are controlled through
    DAQami external to the script. See board_config_6x6.py for similar
    calibration with the lasers controlled through the script.
'''

import vivilux.hardware.daq as daq
import vivilux.hardware.mcc as mcc
import vivilux.hardware.nidaq as ni

ni_boards =[
    ni.USB_6210("NI", 33703915,
        ni.AIPIN("PD_1_0", 0),
        ni.AIPIN("PD_1_1", 1),
        ni.AIPIN("PD_1_2", 2),
        ni.AIPIN("PD_1_3", 3),
        ni.AIPIN("PD_1_4", 4),
        ni.AIPIN("PD_1_5", 5),
        ni.AIPIN("PD_2_0", 6),
        ni.AIPIN("PD_2_1", 7),
        ni.AIPIN("PD_2_2", 8),
        ni.AIPIN("PD_2_3", 9),
        ni.AIPIN("PD_2_4", 10),
        ni.AIPIN("PD_2_5", 11),
        ni.AIPIN("PD_3_0", 12),
        ni.AIPIN("PD_3_1", 13),
        ni.AIPIN("PD_3_2", 14),
        ni.AIPIN("PD_3_3", 15),
    ),
]

mcc_boards = [
    # mcc.USB_1208FS_PLUS("Laser controller", "21E8194",
    #     mcc.AOPIN("Laser_0", 0),
    #     mcc.AOPIN("Laser_1", 1),
    # ),
    mcc.USB_3114("lasers", "1FD2F3C",
        mcc.AOPIN("laser_1", 0),
        mcc.AOPIN("laser_2", 2),
        mcc.AOPIN("laser_3", 4),
        # mcc.AOPIN("laser_4", 6),
        # mcc.AOPIN("laser_5", 1),
        # mcc.AOPIN("laser_6", 3),
    ),
]

# Create a netlist with the NI and MCC boards
ni.config_detected_devices(ni_boards, verbose=False)
mcc.config_detected_devices(mcc_boards, verbose=False)
netlist = daq.Netlist(*ni_boards, *mcc_boards)

if __name__ == "__main__":
    from time import sleep
    from vivilux.hardware.visualization import HeatmapVisualizer
    from vivilux.hardware.detectors import DetectorArray
    import numpy as np
    np.set_printoptions(suppress=True,)    

    detector_nets = ["PD_1_0", "PD_2_0", "PD_3_0",] # may need to rewire for "PD_4_0", "PD_5_0", "PD_6_0",] in one board
    detectors_state = HeatmapVisualizer(shape=(3, 1), vmin=0, vmax=1,
                                        xlabel="Detector Column",
                                        ylabel="Detector Row",
                                        clabel="Photocurrent (uA)")
    
    delay_read = 100e-3 # 100 ms delay before reading the detectors
    
    # Test streaming data from the NI board
    with netlist:
        # initialize the detector array
        detector_array = DetectorArray(3, # 6
                                       nets=detector_nets,
                                       netlist=netlist,
                                       transimpedance=220e3,  # 220k ohms
                                       )
        while True:
            # Read the detector values
            # data = netlist.group_vin(detector_nets)
            data = detector_array.read()
            print(data*1e6)
            detectors_state.update(data.reshape((3, 1))*1e6)
            sleep(delay_read)