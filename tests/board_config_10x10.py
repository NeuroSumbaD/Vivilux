'''This board configuration describes the experiment using the SFP lasers as
    optical sources for the 10x10 MZI mesh. 
'''

import vivilux.hardware.daq as daq
import vivilux.hardware.nidaq as ni
import vivilux.hardware.ser as ser
import vivilux.hardware.mcc as mcc

# ------------ Set up boards and netlist ------------

ser_boards = [
    ser.EVAL_AD5370("DAC0", 0, 17, # 40-channel DAC with buffer board with csPin=17
        *[ser.AOPIN(f"b0_ch_{index}", index) for index in range(40)],
    ),
    ser.EVAL_AD5370("DAC1", 1, 20, # 40-channel DAC with buffer board with csPin=17
        *[ser.AOPIN(f"b1_ch_{index}", index) for index in range(40)],
    ),
]


pico = ser.BoardManager("PICO-002", *ser_boards)

# fpga = ser.VC_709("VC-709",
#                   ser.DIOPIN("Laser_0",0),
#                   ser.DIOPIN("Laser_1",1),
#                   ser.DIOPIN("Laser_2",2),
#                   ser.DIOPIN("Laser_3",3),
#                   )

# Create a netlist with the NI and MCC boards
ser.config_detected_devices([pico,
                            #  fpga,
                             ], verbose=False)
netlist = daq.Netlist(pico,
                    #   fpga,
                      )

if __name__ == "__main__":
    with netlist:
        print("Setting board 0 channel 1 to 5.67...")
        netlist["b1_ch_1"].vout(5.67)
        print("Entering interactive mode, type any valid python command below...")
        breakpoint()
        print("You have now exited the interactive mode, reseting all boards and exiting program...")