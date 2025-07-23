'''This module is intended for plotting and comparing the control voltage
    versus measured
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
    from vivilux.hardware.detectors import DetectorArray
    from vivilux.hardware.lasers import LaserArray
    
    import tqdm  # For progress bar
    import matplotlib.pyplot as plt
    import numpy as np
    np.set_printoptions(suppress=True,)    

    detector_nets = ["PD_1_0", "PD_2_0", "PD_3_0",] # may need to rewire for "PD_4_0", "PD_5_0", "PD_6_0",] in one board
    delay_read = 100e-3 # 100 ms delay before reading the detectors

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    axs[0].set_title("Laser 1 Sweep")
    axs[1].set_title("Laser 2 Sweep")
    axs[2].set_title("Laser 3 Sweep")
    
    num_points = 100
    laser_sweep = np.linspace(0, 1, num_points)  # Normalized power sweep
    current_readings = np.zeros((3, num_points, 3))  # Store readings for each laser

    # Test streaming data from the NI board
    with netlist:
        # Initialize the detector array
        detector_array = DetectorArray(3, # 6
                                       nets=detector_nets,
                                       netlist=netlist,
                                       transimpedance=220e3,  # 220k ohms
                                       )
        
        # Initialize the laser array
        laser_array = LaserArray(3, # 6
                                netlist = netlist,
                                control_nets = ["laser_3", "laser_1", "laser_2",], # "laser_4", "laser_5", "laser_6",]
                                detectors = detector_array,
                                limits=(0, 10),
                                )
        for channel in range(3):
            for index, power in tqdm.tqdm(enumerate(laser_sweep), desc=f"Channel {channel+1} Sweep", total=num_points):
                normal_power = np.zeros(3)
                normal_power[channel] = power
                # Set the laser power for the current channel
                laser_array.setNormalized(normal_power)
                sleep(delay_read)

                current_readings[channel, index] = detector_array.read()*1e6  # Convert to microamperes (uA)
                
                laser_array.setNormalized(np.zeros(3))  # Turn off the lasers after each reading
                sleep(10*delay_read) # 10% duty cycle to allow lasers to cool down

            # Plot the results
            for det_chan in range(3):
                axs[channel].plot(laser_sweep,
                                  current_readings[channel, :, det_chan], 
                                  label=f"Ch{det_chan+1}",
                                  )
            axs[channel].set_xlabel("Laser Power (Normalized)")
            axs[channel].set_ylabel("Photocurrent (uA)")
            axs[channel].legend()
    plt.tight_layout()
    plt.show()