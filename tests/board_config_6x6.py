'''This module defines the netlist for the current board configuration for the
    6x6 MZI mesh on the AIM 2020 tapeout. Running this script by itself will
    initialize the netlist and start streaming data from the NI board, which can
    be used to calibrate the polarization controllers for the laser.
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
    mcc.USB_1208FS_PLUS("ADC 1", "21E81FE",
        mcc.AIPIN("PD_3_4", 0), # Pin 1
        mcc.AIPIN("PD_3_5", 1), # Pin 2
        mcc.AIPIN("PD_4_0", 2), # Pin 4
        mcc.AIPIN("PD_4_1", 3), # Pin 5
        mcc.AIPIN("PD_4_2", 4), # Pin 7
        mcc.AIPIN("PD_4_3", 5), # Pin 8
        mcc.AIPIN("PD_4_4", 6), # Pin 10
        mcc.AIPIN("PD_4_5", 7), # Pin 11
    ),
    mcc.USB_1208FS_PLUS("ADC 2", "21E81BD",
        mcc.AIPIN("PD_5_0", 0), # Pin 1
        mcc.AIPIN("PD_5_1", 1), # Pin 2
        mcc.AIPIN("PD_5_2", 2), # Pin 4
        mcc.AIPIN("PD_5_3", 3), # Pin 5
        mcc.AIPIN("PD_5_4", 4), # Pin 7
        mcc.AIPIN("PD_5_5", 5), # Pin 8
        mcc.AIPIN("PD_6_0", 6), # Pin 10
        mcc.AIPIN("PD_6_1", 7), # Pin 11
    ),
    mcc.USB_1208FS_PLUS("ADC 3", "21E8232",
        mcc.AIPIN("PD_6_2", 0), # Pin 1
        mcc.AIPIN("PD_6_3", 1), # Pin 2
        mcc.AIPIN("PD_6_4", 2), # Pin 4
        mcc.AIPIN("PD_6_5", 3), # Pin 5
    ),
    mcc.USB_3114("DAC", "216B19F",
        mcc.AOPIN("1_1_i", 1),
        mcc.AOPIN("1_3_i", 5),
        mcc.AOPIN("1_3_o", 7),
        mcc.AOPIN("1_5_i", 2),
        mcc.AOPIN("2_2_i", 3),
        mcc.AOPIN("2_4_i", 4),
        mcc.AOPIN("3_1_i", 6),
        mcc.AOPIN("3_3_i", 8),
        mcc.AOPIN("3_5_i", 10),
        mcc.AOPIN("4_2_i", 0),
        mcc.AOPIN("4_4_i", 15),
        mcc.AOPIN("5_1_i", 9),
        mcc.AOPIN("5_3_i", 11),
        mcc.AOPIN("5_5_i", 13),
    ),
    mcc.USB_3114("lasers", "000000",
        mcc.AOPIN("laser_1", 1),
        mcc.AOPIN("laser_2", 2),
        mcc.AOPIN("laser_3", 3),
        # mcc.AOPIN("laser_4", 3),
        # mcc.AOPIN("laser_5", 4),
        # mcc.AOPIN("laser_6", 5),
    ),
]

# Create a netlist with the NI and MCC boards
ni.config_detected_devices(ni_boards, verbose=False)
mcc.config_detected_devices(mcc_boards, verbose=False)
netlist = daq.Netlist(*ni_boards, *mcc_boards)

if __name__ == "__main__":
    # Running this script as main will plot the detector readings live so they
    # can be used to calibrate the polarization controllers for the laser.
    from time import sleep
    from vivilux.hardware.visualization import HeatmapVisualizer
    from vivilux.hardware.lasers import LaserArray
    from vivilux.hardware.detectors import DetectorArray

    detector_nets = ["PD_1_0", "PD_2_0", "PD_3_0",] # may need to rewire for "PD_4_0", "PD_5_0", "PD_6_0",] in one board
    detectors_state = HeatmapVisualizer(shape=(3, 1), vmin=0, vmax=2,
                                        xlabel="Detector Column",
                                        ylabel="Detector Row",
                                        clabel="Voltage")
    
    # Initialize the laser array
    laser_array = LaserArray(3, # 6
                             netlist = netlist,
                             control_nets = ["laser_1", "laser_2", "laser_3",], # "laser_4", "laser_5", "laser_6",]
                             detector_nets = DetectorArray(3, # 6
                                                           nets=detector_nets,
                                                           netlist=netlist,
                                                           transimpedance=220e3,  # 220k ohms
                                                           ),
                             limits=(0, 10),
                             )
    
    delay_read = 100e-3 # 100 ms delay before reading the detectors
    
    # Test streaming data from the NI board
    with netlist:
        while True:
            # Set the laser power to 1.0 for all lasers
            laser_array.setNormalized([1.0, 1.0, 1.0])  # Adjust as needed for more lasers
            sleep(delay_read) # wait for the lasers to stabilize

            # Read the detector values
            data = netlist.group_vin(detector_nets)
            print(data)
            detectors_state.update(data.reshape((3, 1)))

            # Turn off the lasers
            laser_array.setNormalized([0.0, 0.0, 0.0])

            sleep(10*delay_read) # 10% duty cycle for the lasers