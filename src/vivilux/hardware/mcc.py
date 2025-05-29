'''Submodule providing interface for Measurement Computing (MCC) DAQ hardware.
'''

from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from mcculw.enums import InterfaceType

from vivilux.logger import log

class BoardInfo:
    def __init__(self, board_num: int, daq_dev_info: ul.DaqDeviceDescriptor):
        self.board_num = board_num
        self.daq_dev_info = daq_dev_info

    def __repr__(self):
        return f'BoardInfo(board_num={self.board_num}, ' \
               f'daq_dev_info={self.daq_dev_info.product_name} ({self.daq_dev_info.unique_id}))'

class BoardConfig:
    def __init__(self):
        self.boards: dict[str, BoardInfo] = {}

    def add_board(self, board_num: int, daq_dev_info: ul.DaqDeviceDescriptor):
        """Adds a board to the UL with the specified board number and device info."""
        ul.create_daq_device(board_num, daq_dev_info)
        print(f'Board {board_num} configured with device: {daq_dev_info.product_name} ({daq_dev_info.unique_id})')


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

def config_detected_devices(device_ids: list[str],
                            verbose: bool=True,
                            ) -> dict[str, int]:
    '''Search the connected DAQ devices and add them to the UL with an assigned
        board number based on their position in the device_ids list.

        Parameters
        ----------
        device_ids : list[str]
            A list of unique IDs corresponding to the ID under the physical board.add()
        verbose : bool, optional
            If True, prints the found devices and their IDs, else . Default is True.

        Returns
        -------
        dict[str, BoardConfig]
            A dictionary mapping unique device IDs to their BoardConfig objects.

        TODO: implement logger (logger submodule should initialize logger and provide global access to it?)
    '''
    
    ul.ignore_instacal()
    devices = ul.get_daq_device_inventory(InterfaceType.ANY)
    if not devices:
        raise Exception('Error: No DAQ devices found')
    
    board_nums: dict[str, BoardConfig] = {}
    
    if verbose:
        print('Found', len(devices), 'DAQ device(s):')
        for device in devices:
            print(f'Name: {device.product_name} ({device.unique_id}) - '
                  f'Device ID = {device.product_id}')
            if device.unique_id in device_ids:
                index = device_ids.index(device.unique_id)
                ul.create_daq_device(index, device)
                board_nums[device.unique_id] = BoardConfig(index, device)
    else:
        for device in devices:
            if device.unique_id in device_ids:
                index = device_ids.index(device.unique_id)
                ul.create_daq_device(index, device)
                board_nums[device.unique_id] = BoardConfig(index, device)

    return board_nums
    
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

class PIN_DIRECTION:
    INPUT = 'I'
    OUTPUT = 'O'
    IO = 'IO'  # Input/Output, used for digital pins

class PIN_TYPE:
    ANALOG = 'A'  # Analog pin, used for analog pins
    DIGITAL = 'D'  # Digital pin, used for digital pins

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
            log.error(f"Invalid channel {self.chnl}")
            raise Exception(f"Invalid channel {self.chnl}")
        return port, int(self.chnl[1:])
    

class DIOPIN(PIN):
    direction = PIN_DIRECTION.IO
    type = PIN_TYPE.DIGITAL

    def __init__(self, dev, chnl, digital=True, type='IO'):
        super().__init__(dev, chnl, digital, type)
        if not self.digital:
            log.error(f"Digital pin {self.chnl} cannot be used as an analog pin")
            raise Exception(f"Digital pin {self.chnl} cannot be used as an analog pin")
        
class AOPIN(PIN):
    direction = PIN_DIRECTION.OUTPUT
    type = PIN_TYPE.ANALOG

    def __init__(self, dev, chnl, digital=False, type='O'):
        super().__init__(dev, chnl, digital, type)
        if self.digital:
            log.error(f"Analog pin {self.chnl} cannot be used as a digital pin")
            raise Exception(f"Analog pin {self.chnl} cannot be used as a digital pin")
   
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
log.info("New session started.")
log.info(f'Found {len(devices)} DAQ device(s):')
for board_num, device in enumerate(devices):
    log.info(f'  {device.product_name} ({device.unique_id}) - '
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
        log.error(f"Pin {pinName} not found in the channel map")
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
        log.error(f"Pin {pinName} is not an output pin")
        raise Exception(f"Pin {pinName} is not an output pin")
    
    if pin.digital:
        log.error(f"Pin {pinName} is a digital pin")
        raise Exception(f"Pin {pinName} is a digital pin")
    
    log.debug(f"Set pin {pinName} to {voltage}V")
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
        log.error(f"Pin {pinName} is not an input pin")
        raise Exception(f"Pin {pinName} is not an input pin")
    
    if pin.digital:
        log.error(f"Pin {pinName} is a digital pin")
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
        log.error(f"Pin {pinName} is not an output pin")
        raise Exception(f"Pin {pinName} is not an output pin")
    
    if not pin.digital:
        log.error(f"Pin {pinName} is an analog pin")
        raise Exception(f"Pin {pinName} is an analog pin")
    
    port, channel = pin.DIO_chan()
    
    # get the current integer representation
    x = board_digital_ports[pin.dev][pin.chnl[0]]
    
    # set the bit to 0
    x &= ~(1 << channel)
    # set the bit to the value
    x = (x & ~(1 << channel)) | ((value & 1) << channel)
    board_digital_ports[pin.dev][pin.chnl[0]] = x
    
    log.debug(f"Set pin {pinName} to {value} by writing "
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