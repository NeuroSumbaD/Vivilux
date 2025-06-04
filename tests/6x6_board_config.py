import vivilux.hardware.daq as daq
import vivilux.hardware.mcc as mcc
import vivilux.hardware.nidaq as ni

ni_boards =[
    ni.USB_6210("NI", 33703915,
        ni.AIPIN("PD_1_0", "A0"),
        ni.AIPIN("PD_1_0", "A0"),
        ni.AIPIN("PD_1_1", "A1"),
        ni.AIPIN("PD_1_2", "A2"),
        ni.AIPIN("PD_1_3", "A3"),
        ni.AIPIN("PD_1_4", "A4"),
        ni.AIPIN("PD_1_5", "A5"),
        ni.AIPIN("PD_2_0", "A6"),
        ni.AIPIN("PD_2_1", "A7"),
        ni.AIPIN("PD_2_2", "A8"),
        ni.AIPIN("PD_2_3", "A9"),
        ni.AIPIN("PD_2_4", "A10"),
        ni.AIPIN("PD_2_5", "A11"),
        ni.AIPIN("PD_3_0", "A12"),
        ni.AIPIN("PD_3_1", "A13"),
        ni.AIPIN("PD_3_2", "A14"),
        ni.AIPIN("PD_3_3", "A15"),
    ),
]

mcc_boards = [
    mcc.USB_1208FS_PLUS("ADC FE", "",
        mcc.AIPIN("PD_3_4", 0), # Pin 1
        mcc.AIPIN("PD_3_5", 1), # Pin 2
        mcc.AIPIN("PD_4_0", 2), # Pin 4
        mcc.AIPIN("PD_4_1", 3), # Pin 5
        mcc.AIPIN("PD_4_2", 4), # Pin 7
        mcc.AIPIN("PD_4_3", 5), # Pin 8
        mcc.AIPIN("PD_4_4", 6), # Pin 10
        mcc.AIPIN("PD_4_5", 7), # Pin 11
    ),
    mcc.USB_1208FS_PLUS("ADC BD", "",
        mcc.AIPIN("PD_5_0", 0), # Pin 1
        mcc.AIPIN("PD_5_1", 1), # Pin 2
        mcc.AIPIN("PD_5_2", 2), # Pin 4
        mcc.AIPIN("PD_5_3", 3), # Pin 5
        mcc.AIPIN("PD_5_4", 4), # Pin 7
        mcc.AIPIN("PD_5_5", 5), # Pin 8
        mcc.AIPIN("PD_6_0", 6), # Pin 10
        mcc.AIPIN("PD_6_1", 7), # Pin 11
    ),
    mcc.USB_1208FS_PLUS("ADC 32", "",
        mcc.AIPIN("PD_6_2", 0), # Pin 1
        mcc.AIPIN("PD_6_3", 1), # Pin 2
        mcc.AIPIN("PD_6_4", 2), # Pin 4
        mcc.AIPIN("PD_6_5", 3), # Pin 5
    ),
    mcc.USB_3114("DAC 9F", "216B19F",
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
    )
]

netlist = daq.Netlist(*ni_boards, *mcc_boards)