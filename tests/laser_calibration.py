'''This module is a configuration intended for calibrating the laser
    polarization controllers on the 6x6 MZI mesh on the AIM 2020 tapeout.

    TODO: Formalize the calibration procedure and detail the instructions
    within this module.
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
    # mcc.USB_1208FS_PLUS("Laser controller", "",
    #     mcc.AOPIN("Laser_0", 0),
    #     mcc.AOPIN("Laser_1", 1),
    # ),
]

# Create a netlist with the NI and MCC boards
ni.config_detected_devices(ni_boards, verbose=False)
# mcc.config_detected_devices(mcc_boards, verbose=False)
netlist = daq.Netlist(*ni_boards, *mcc_boards)

if __name__ == "__main__":
    from time import sleep
    from vivilux.hardware.visualization import HeatmapVisualizer

    detector_nets = ["PD_1_0", "PD_2_0", "PD_3_0",] # may need to rewire for "PD_4_0", "PD_5_0", "PD_6_0",] in one board
    detectors_state = HeatmapVisualizer(shape=(3, 1), vmin=0, vmax=2,
                                        xlabel="Detector Column",
                                        ylabel="Detector Row",
                                        clabel="Voltage")
    
    # Test streaming data from the NI board
    with netlist:
        while True:
            data = netlist.group_vin(detector_nets)
            print(data)
            detectors_state.update(data.reshape((3, 1)))
            sleep(0.5)