import vivilux.hardware.daq as daq
import vivilux.hardware.ser as ser

if __name__=="__main__":
    dac = ser.EVAL_AD5370("DAC1",   # name
                      0,        # uid
                      17,       # csPin
                      ser.AOPIN("1_1_i", 0),
                      )

    pico = ser.BoardManager("PICO-001", # uid
                        dac,
                        # bad_dac1,
                        # bad_dac2,
                        )
    
    fpga = ser.VC_709("VC-709",
                      ser.DIOPIN("Laser 0",0),
                      ser.DIOPIN("Laser 1",1),
                      ser.DIOPIN("Laser 2",2),
                      ser.DIOPIN("Laser 3",3),
                      )
    
    ser.config_detected_devices([pico, fpga], verbose=True)
    netlist = daq.Netlist(pico, fpga)
    with netlist:
        print("Turning on laser 1...")
        netlist["Laser 1"].dout(True)
        fpga.update_vibration([1,1,1,1])
        print(f"Input binary laser states below (i.e. '1010')")
        while (True):
            states = input(f"Current state = {fpga.laser_states}, new state ('q' to exit): ")
            if states.lower() in ['q', 'quit', 'exit']:
                print("Exiting...")
                break
            else:
                fpga.update_lasers([int(bit) for bit in states])
