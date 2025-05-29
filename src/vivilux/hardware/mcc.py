'''Submodule providing interface for Measurement Computing (MCC) DAQ hardware.
'''

from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from mcculw.enums import InterfaceType

def config_first_detected_device(board_num, dev_id_list=None):
    """Adds the first available device to the UL.  If a types_list is specified,
    the first available device in the types list will be add to the UL.

    Parameters
    ----------
    board_num : int
        The board number to assign to the board when configuring the device.

    dev_id_list : list[int], optional
        A list of product IDs used to filter the results. Default is None.
        See UL documentation for device IDs.
    """
    ul.ignore_instacal()
    devices = ul.get_daq_device_inventory(InterfaceType.ANY)
    if not devices:
        raise Exception('Error: No DAQ devices found')

    print('Found', len(devices), 'DAQ device(s):')
    for device in devices:
        print('  ', device.product_name, ' (', device.unique_id, ') - ',
              'Device ID = ', device.product_id, sep='')

    device = devices[0]
    if dev_id_list:
        device = next((device for device in devices
                       if device.product_id in dev_id_list), None)
        if not device:
            err_str = 'Error: No DAQ device found in device ID list: '
            err_str += ','.join(str(dev_id) for dev_id in dev_id_list)
            raise Exception(err_str)
            
            

    # Add the first DAQ device to the UL with the specified board number
    ul.create_daq_device(0, devices[0])
    ul.create_daq_device(1, devices[1])
    ul.create_daq_device(2, devices[2])
    #ul.create_daq_device(3, devices[3])
    
# DAC initialization
def DAC_Init():
    use_device_detection = True
    dev_id_list = []
    board_num = 0
     
    try:
        if use_device_detection:
            config_first_detected_device(board_num, dev_id_list)
    
        daq_dev_info = DaqDeviceInfo(board_num)
        if not daq_dev_info.supports_analog_output:
            raise Exception('Error: The DAQ device does not support '
                            'analog output')
    
        print('\nActive DAQ device: ', daq_dev_info.product_name, ' (',
              daq_dev_info.unique_id, ')\n', sep='')
    
        ao_info = daq_dev_info.get_ao_info()
        ao_range = ao_info.supported_ranges[0]
        return ao_range
    except Exception as e:
        print('\n', e)

ao_range = DAC_Init()

from mcculw.enums import InterfaceType
import mcculw.ul as ul
from mcculw.device_info import DaqDeviceInfo
from mcculw.enums import (AnalogInputMode, DigitalPortType, DigitalIODirection)

import logging
import os
import __main__

# Set up logging
def setup_logger():
    filename = os.path.basename(__main__.__file__)
    filename = filename.replace('.py', '.log')
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    
    # Check if handler already exists to avoid duplicates
    if not logger.handlers:
        # Create file handler and set formatter
        file_handler = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# initialize logger
logger = setup_logger()

# Fetch boards and their IDs
ul.ignore_instacal()
devices = ul.get_daq_device_inventory(InterfaceType.ANY)
devsIDsList = [device.unique_id for device in devices]
## 
# Variables, Functions, and Classes. Note: keep the order of what's defined here.
board_names = {0: "21E8245",
              1: "21E8230",
              2: "21ED703",
              }

board_digital_ports = {
    "21E8245": {"A": 0b0, "B": 0b0},
    "21E8230": {"A": 0b0, "B": 0b0},
    "21ED703": {"A": 0b0, "B": 0b0},
}

def devFromList(devData):
    if type(devData) == int:
        return devsIDsList.index(board_names[devData])
    return devsIDsList.index(devData)

class PIN:
    def __init__(self, dev, chnl, digital=True, type='IO'):
        self.dev = dev
        self.chnl = chnl
        self.digital = digital
        self.type = type
    @property
    def devNum(self):
        return devFromList(self.dev)
    def DIO_chan(self):
        '''Returns the port and channel number for the digital pin
        '''
        if "A" in self.chnl:
            port = DigitalPortType.FIRSTPORTA
        elif "B" in self.chnl:
            port = DigitalPortType.FIRSTPORTB
        else:
            logger.error(f"Invalid channel {self.chnl}")
            raise Exception(f"Invalid channel {self.chnl}")
        return port, int(self.chnl[1:])
    
# channelMap = {
#               # Digital IO pins
#               "IN_G1_L": PIN('21E8230', 39, digital=True, type='IO'),
#               "IN_G2_L": PIN('21E8230', 38, digital=True, type='IO'),
#               "IN_G3_L": PIN('21E8230', 37, digital=True, type='IO'),
#               "IN_G4_L": PIN('21ED703', 39, digital=True, type='IO'),
#               "IN_G5_L": PIN('21E8245', 39, digital=True, type='IO'),
#               "IN_G6_L": PIN('21E8245', 38, digital=True, type='IO'),
#               "IN_G7_L": PIN('21ED703', 38, digital=True, type='IO'),
#               "IN_G8_L": PIN('21E8245', 37, digital=True, type='IO'),
#               "IN_G9_L": PIN('21E8245', 36, digital=True, type='IO'),
#               "IN_S": PIN('21E8230', 36, digital=True, type='IO'),
#               "GND_S": PIN('21ED703', 37, digital=True, type='IO'),
#               # Analog Write
#               "V1_W_1": PIN('21E8230', 14, digital=False, type='O'),
#               "V2_W_1": PIN('21ED703', 14, digital=False, type='O'),
#               "V3_W_1": PIN('21E8245', 14, digital=False, type='O'),
#               "V1_R_1": PIN('21E8230', 13, digital=False, type='O'),
#               "V2_R_1": PIN('21ED703', 13, digital=False, type='O'),
#               "V3_R_1": PIN('21E8245', 13, digital=False, type='O'),
#               # Analog Read
#               "Y1_1": PIN('21E8245', 11, digital=False, type='I'),
#               "Y2_1": PIN('21E8245', 10, digital=False, type='I'),
#               "Z1_1": PIN('21E8245', 8, digital=False, type='I'),
#               }

channelMap = {
              # Digital IO pins
              "IN_G1_L": PIN('21E8230', "B7", digital=True, type='IO'),
              "IN_G2_L": PIN('21E8230', "B6", digital=True, type='IO'),
              "IN_G3_L": PIN('21E8230', "B5", digital=True, type='IO'),
              "IN_G4_L": PIN('21ED703', "B7", digital=True, type='IO'),
              "IN_G5_L": PIN('21E8245', "B7", digital=True, type='IO'),
              "IN_G6_L": PIN('21E8245', "B6", digital=True, type='IO'),
              "IN_G7_L": PIN('21ED703', "B6", digital=True, type='IO'),
              "IN_G8_L": PIN('21E8245', "B5", digital=True, type='IO'),
              "IN_G9_L": PIN('21E8245', "B4", digital=True, type='IO'),
              "IN_S": PIN('21E8230', "B4", digital=True, type='IO'),
              "GND_S": PIN('21ED703', "B5", digital=True, type='IO'),
              "V1_W_so": PIN('21ED703', "B4", digital=True, type='IO'),
              "V2_W_so": PIN('21ED703', "B3", digital=True, type='IO'),
              "V3_W_so": PIN('21ED703', "B2", digital=True, type='IO'),
              # Analog Write
              "V1_W_1": PIN('21E8230', 1, digital=False, type='O'),
              "V2_W_1": PIN('21ED703', 1, digital=False, type='O'),
              "V3_W_1": PIN('21E8245', 1, digital=False, type='O'),
              "V1_R_1": PIN('21E8230', 0, digital=False, type='O'),
              "V2_R_1": PIN('21ED703', 0, digital=False, type='O'),
              "V3_R_1": PIN('21E8245', 0, digital=False, type='O'),
              # Analog Read
              "Y1_1": PIN('21E8245', 7, digital=False, type='I'),
              "Y2_1": PIN('21E8245', 6, digital=False, type='I'),
              "Z1_1": PIN('21E8245', 5, digital=False, type='I'),
              }

##
# Continue initialization
logger.info("New session started.")
logger.info(f'Found {len(devices)} DAQ device(s):')
for board_num, device in enumerate(devices):
    logger.info(f'  {device.product_name} ({device.unique_id}) - '
        f'Device ID = {device.product_id}')
    ul.create_daq_device(board_num, device)
    if device.product_name == "USB-1208FS-Plus":
        ul.a_input_mode(board_num, AnalogInputMode.SINGLE_ENDED)

daq_dev_info = DaqDeviceInfo(devFromList('21E8245'))
ao_info = daq_dev_info.get_ao_info()
ao_range = ao_info.supported_ranges[0]
ai_info = daq_dev_info.get_ai_info()
ai_range = ai_info.supported_ranges[0]
##

# Config Digital IO Pins
for pinName, pin in channelMap.items():
    if pin.digital:
        port, channel = pin.DIO_chan()
        # ul.d_config_bit(pin.devNum, port, channel, DigitalIODirection.OUT) # NOT BIT ADDRESSIBLE
        ul.d_config_port(pin.devNum, port, DigitalIODirection.OUT)

def getPin(pinName:str):
    if pinName not in channelMap:
        logger.error(f"Pin {pinName} not found in the channel map")
        raise Exception(f"Pin {pinName} not found in the channel map")
    return channelMap[pinName]


def vout(pinName:str, voltage:float): # this function is all you need now. just use the pin name and voltage vale
    '''
    Sets the voltage of the specified pin
    Parameters:
    pinName: str: The name of the pin to set the voltage of
    voltage: float: The voltage to set the pin to
    '''
    pin = getPin(pinName)
    
    if "O" not in pin.type.upper():
        logger.error(f"Pin {pinName} is not an output pin")
        raise Exception(f"Pin {pinName} is not an output pin")
    
    if pin.digital:
        logger.error(f"Pin {pinName} is a digital pin")
        raise Exception(f"Pin {pinName} is a digital pin")
    
    logger.debug(f"Set pin {pinName} to {voltage}V")
    ul.v_out(pin.devNum, pin.chnl, ao_range, voltage)
    return

def vin(pinName:str):
    '''
    Reads the voltage from the specified pin
    Parameters:
    pinName: str: The name of the pin to read from
    '''
    pin = getPin(pinName)
    
    if "I" not in pin.type.upper():
        logger.error(f"Pin {pinName} is not an input pin")
        raise Exception(f"Pin {pinName} is not an input pin")
    
    if pin.digital:
        logger.error(f"Pin {pinName} is a digital pin")
        raise Exception(f"Pin {pinName} is a digital pin")
    
    return ul.v_in(pin.devNum, pin.chnl, ai_range)

def dout(pinName:str, value:bool):
    '''
    Sets the digital value of the specified pin
    Parameters:
    pinName: str: The name of the pin to set the digital value of
    value: bool: The digital value to set the pin to
    '''
    pin = getPin(pinName)
    
    if "O" not in pin.type.upper():
        logger.error(f"Pin {pinName} is not an output pin")
        raise Exception(f"Pin {pinName} is not an output pin")
    
    if not pin.digital:
        logger.error(f"Pin {pinName} is an analog pin")
        raise Exception(f"Pin {pinName} is an analog pin")
    
    port, channel = pin.DIO_chan()
    
    # get the current integer representation
    x = board_digital_ports[pin.dev][pin.chnl[0]]
    
    # set the bit to 0
    x &= ~(1 << channel)
    # set the bit to the value
    x = (x & ~(1 << channel)) | ((value & 1) << channel)
    board_digital_ports[pin.dev][pin.chnl[0]] = x
    
    logger.debug(f"Set pin {pinName} to {value} by writing "
                 f"{format(x, "08b")} to port {pin.chnl[0]}")
    ul.d_out(pin.devNum, port, board_digital_ports[pin.dev][pin.chnl[0]])
    # ul.d_bit_out(pin.devNum, port, channel, int(value)) # NOT BIT ADDRESSIBLE (DOESN'T WORK)
    
def reset_board():
    for pinName, pin in channelMap.items():
        if pin.digital:
            dout(pinName, False)
            # port, channel = pin.DIO_chan()
            # ul.d_out(pin.devNum, port, 0)
        else:
            if pin.type == "O":
                vout(pinName, 0)
                # raw_value = ul.from_eng_units(self.board_num, ao_range, data_value)
                # ul.a_out(pin.devNum, pin.chnl, ao_range, 0)
    return

device_grid = [["IN_G1_L", "IN_G2_L", "IN_G3_L"],
               ["IN_G4_L", "IN_G5_L", "IN_G6_L"],
               ["IN_G7_L", "IN_G8_L", "IN_G9_L"],]

input_rows = ["V1_R_1", "V2_R_1", "V3_R_1"]

gate_rows = ["V1_W_1", "V2_W_1", "V3_W_1"]

output_cols = ["Y1_1", "Y2_1", "Z1_1"]