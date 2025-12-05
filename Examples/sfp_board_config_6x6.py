'''This board configuration describes the experiment using the SFP lasers as
    optical sources for the 6x6 MZI mesh. Currently only 4 lasers are plugged in,
    but the expansion to larger numbers of lasers will be handled soon.
'''

import vivilux.hardware.daq as daq
import vivilux.hardware.nidaq as ni
import vivilux.hardware.ser as ser
# import vivilux.hardware.mcc as mcc

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
        num_samples = 50,
        num_samples_to_skip = 5,
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
        ser.AOPIN("1_5_o", 37),
        
        ser.AOPIN("4_1_i", 14),
        ser.AOPIN("5_2_i", 16),
        ser.AOPIN("4_3_i", 18),
        ser.AOPIN("5_4_i", 20),
        ser.AOPIN("4_5_i", 22),
        
        ser.AOPIN("3_6_i", 39),
        
        ser.AOPIN("2_6_o", 24),
        ser.AOPIN("2_6_i", 26),
        ser.AOPIN("6_1_i", 28),
        ser.AOPIN("6_5_i", 30),
        
        ser.AOPIN("6_5_o", 32),
        ser.AOPIN("6_4_o", 34),
        ser.AOPIN("6_3_o", 36),
        ser.AOPIN("6_3_i", 38),
    ),
]

# mcc_boards = [
#     mcc.USB_3114("modulators", "1FD2F3C",
#         mcc.AOPIN("Mod_0", 0),
#         mcc.AOPIN("Mod_1", 2),
#         mcc.AOPIN("Mod_2", 4),
#         mcc.AOPIN("Mod_3", 6),
#         # mcc.AOPIN("laser_5", 1),
#         # mcc.AOPIN("laser_6", 3),
#     ),
#     mcc.USB_3114("EXTRA", "1FD2F3C",
#         mcc.AOPIN("", 0),
#         mcc.AOPIN("", 0),
#         mcc.AOPIN("", 0),
#         mcc.AOPIN("", 0),
#         mcc.AOPIN("", 0),
#     ),
# ]

pico = ser.BoardManager("PICO-001", *ser_boards)

fpga = ser.VC_709("VC-709-6x6",
                  ser.DIOPIN("Laser_0",0),
                  ser.DIOPIN("Laser_1",1),
                  ser.DIOPIN("Laser_2",2),
                  ser.DIOPIN("Laser_3",3),
                  )

# Create a netlist with the NI and MCC boards
ser.config_detected_devices([pico, fpga], verbose=False)
ni.config_detected_devices(ni_boards, verbose=False)
# mcc.config_detected_devices(mcc_boards, verbose=False)
netlist = daq.Netlist(*ni_boards,
                    #   *mcc_boards,
                      pico, 
                      fpga)