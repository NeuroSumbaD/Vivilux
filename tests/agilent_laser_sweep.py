'''In this module, I am sweeping the laser controls for the Agilent lasers to
    test their interface.
'''

from board_config_6x6 import netlist

if __name__ == "__main__":
    from time import sleep
    from vivilux.hardware.detectors import DetectorArray
    from vivilux.hardware.lasers import LaserArray, AgilentLaserArray, AgilentDetectorArray, dBm_to_mW
    
    import tqdm  # For progress bar
    import matplotlib.pyplot as plt
    import numpy as np
    np.set_printoptions(suppress=True,)    

    detector_nets = ["PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0",]

    fig, axs = plt.subplots(1, 4, figsize=(10, 6))
    axs[0].set_title("Laser 1 Sweep")
    axs[1].set_title("Laser 2 Sweep")
    axs[2].set_title("Laser 3 Sweep")
    axs[3].set_title("Laser 4 Sweep")

    num_points = 30
    laser_sweep = np.linspace(0, 1, num_points)  # Normalized power sweep
    current_readings = np.zeros((4, num_points, 4))  # Store readings for each laser

    # Test streaming data from the NI board
    with netlist:
        # Initialize the detector array
        base_detector_array = DetectorArray(4, # 6
                                       nets=detector_nets,
                                       netlist=netlist,
                                       transimpedance=220e3,  # 220k ohms
                                       )

        # Initialize and calibrate the laser array
        laser_array = AgilentLaserArray(size=4,
                                        detectors=base_detector_array,
                                        upperLimits = dBm_to_mW(np.array([-5, -3, -5, -9.4])),
                                        lowerLimits = dBm_to_mW(np.array([-10, -10, -10, -10])),
                                        port = 'GPIB0::20::INSTR',
                                        channels = [1, 2, 4, 3],
                                        pause = 0.1,
                                        wait = 10,
                                        max_retries = 10,
                                        )

        detector_array = AgilentDetectorArray(detectors=base_detector_array,
                                              lasers=laser_array)
        
        for channel in range(4):
            for index, power in tqdm.tqdm(enumerate(laser_sweep), desc=f"Channel {channel+1} Sweep", total=num_points):
                normal_power = np.zeros(4)
                normal_power[channel] = power
                # Set the laser power for the current channel
                laser_array.setNormalized(normal_power)
                # sleep(delay_read)

                current_readings[channel, index] = detector_array.read()*1e6  # Convert to microamperes (uA)
                
                laser_array.setNormalized(np.zeros(4))  # Turn off the lasers after each reading
                # sleep(10*delay_read) # 10% duty cycle to allow lasers to cool down

            # Plot the results
            for det_chan in range(4):
                axs[channel].plot(laser_sweep * 10 / 480, # convert to real units
                                  current_readings[channel, :, det_chan]*100, # Account for 1% tap
                                  label=f"Ch{det_chan+1}",
                                  )
            axs[channel].set_xlabel("Laser Control (mA)")
            axs[channel].set_ylabel("Estimated Photocurrent from 1% tap (uW)")
            axs[channel].legend()
    # plt.tight_layout()
    plt.show()