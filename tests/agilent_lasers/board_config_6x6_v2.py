'''Second version of the board configuration using the 40-channel DAC and buffer
    board to provide more available channels for the 6x6 MZI optelligent PCB.
    Currently using Agilent 8164A lasers so the net names do not appear.
'''
import vivilux.hardware.daq as daq
import vivilux.hardware.ser as ser
import vivilux.hardware.nidaq as ni

ni_boards =[
    ni.USB_6210("NI", 33703915, # NOTE: ni board is currently the only one with the necessary range for single-ended readout
        ni.AIPIN("PD_1_0", 0), # DON'T CHANGE
        ni.AIPIN("PD_1_5", 5), # DON'T CHANGE
        ni.AIPIN("PD_2_0", 6), # DON'T CHANGE
        ni.AIPIN("PD_2_5", 11), # DON'T CHANGE
        ni.AIPIN("PD_3_0", 12), # DON'T CHANGE
        
        ni.AIPIN("PD_4_0", 1),
        ni.AIPIN("PD_5_0", 2),
        ni.AIPIN("PD_6_0", 3),
        
        ni.AIPIN("PD_3_5", 4),
        ni.AIPIN("PD_4_5", 7),
        ni.AIPIN("PD_5_5", 8),
        ni.AIPIN("PD_6_5", 9),
        
        # ni.AIPIN("PD_1_1", 10), # MOVED
        ni.AIPIN("PD_1_2", 10), # FROM MIDDLE mcc.AIPIN("PD_1_2", 0)
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

# Create a netlist with the NI and MCC boards
ni.config_detected_devices(ni_boards, verbose=False)
ser.config_detected_devices([pico], verbose=False)
netlist = daq.Netlist(*ni_boards, pico)