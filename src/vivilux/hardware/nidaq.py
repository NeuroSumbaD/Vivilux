'''Submodule providing interface to National Instruments DAQ hardware.
'''    

import nidaqmx
import nidaqmx.system


# ADC initialization
system = nidaqmx.system.System.local()
print("Driver version: ", system.driver_version)
for device in system.devices:
    print("Device:\n", device) 