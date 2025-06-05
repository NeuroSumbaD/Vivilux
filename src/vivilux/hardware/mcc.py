'''Submodule providing interface for Measurement Computing (MCC) DAQ hardware.
'''

from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from mcculw import enums
# from mcculw.enums import (AnalogInputMode, DigitalPortType,  DigitalIODirection, InterfaceType)

from vivilux.logger import log
import vivilux.hardware.daq as daq

class Board(daq.Board):
    '''Base class for MCC DAQ boards. Inherits from daq.Board.'''
    
    def __init__(self, name: str, unique_id: str, *pins: list[daq.PIN]):
        super().__init__(name, *pins)
        self.unique_id = unique_id
        self.board_num = None  # Will be set when the board is initialized
        self.analog_input_mode = None  # To be defined in subclasses if needed
        
        self.daq_dev_info = None
        self.ao_info = None
        self.ao_range = None
        self.ai_info = None
        self.ai_range = None

    def initialize_ports(self):
        '''Initialize the ports for the board. To be implemented in subclasses.'''
        raise NotImplementedError("Subclasses must implement this method.")
    
    def getBoardInfo(self):
        self.daq_dev_info = DaqDeviceInfo(self.board_num)
        self.ao_info = self.daq_dev_info.get_ao_info()
        self.ao_range = self.ao_info.supported_ranges[0] if self.ao_range is None else self.ao_range
        self.ai_info = self.daq_dev_info.get_ai_info()
        self.ai_range = self.ai_info.supported_ranges[0] if self.ao_range is None else self.ao_range

    def reset(self):
        '''Reset the board by clearing all outputs and setting analog outputs to 0V.'''
        for pin in self.pins.values():
            pin.reset()

def get_detected_devices():
    '''Search the connected DAQ devices and return a list of their unique IDs.

        Returns
        -------
        list[str]
            A list of unique IDs corresponding to the detected DAQ devices.
    '''
    
    ul.ignore_instacal()
    devices: list[ul.DaqDeviceDescriptor] = ul.get_daq_device_inventory(enums.InterfaceType.ANY)
    if not devices:
        raise Exception('Error: No DAQ devices found')
    
    device_ids = [device.unique_id for device in devices]
    
    print('Found', len(devices), 'DAQ device(s):')
    for device in devices:
        print(f'Name: {device.product_name} ({device.unique_id}) - '
              f'Device ID = {device.product_id}')

def config_detected_devices(boards: list[Board],
                            verbose: bool=True,
                            ):
    '''Search the connected DAQ devices and add them to the UL with an assigned
        board number based on their position in the device_ids list.

        Parameters
        ----------
        boards : list[Board]
            A list of Board objects to be configured with the detected devices.
            Each board should have a unique ID that matches the device's serial number.
        verbose : bool, optional
            If True, prints the found devices and their IDs, else . Default is True.

        Returns
        -------
        dict[str, BoardConfig]
            A dictionary mapping unique device IDs to their BoardConfig objects.
    '''
    
    ul.ignore_instacal()
    devices: list[ul.DaqDeviceDescriptor] = ul.get_daq_device_inventory(enums.InterfaceType.ANY)
    if not devices:
        raise Exception('Error: No MCC DAQ devices found')
    
    device_ids = [board.unique_id for board in boards]

    num_boards = len(boards)
    num_initialized = 0
    
    if verbose:
        print('Found', len(devices), 'MCC DAQ device(s):')
        for device in devices:
            print(f'Name: {device.product_name} ({device.unique_id}) - '
                  f'Device ID = {device.product_id}')
            if device.unique_id in device_ids:
                index = device_ids.index(device.unique_id)
                ul.create_daq_device(index, device)
                boards[index].board_num = index
                boards[index].initialize_ports()
                num_initialized += 1
    else:
        log.info(f'Found {len(devices)} MCC DAQ device(s):')
        for device in devices:
            log.info(f'Name: {device.product_name} ({device.unique_id}) - '
                  f'Device ID = {device.product_id}')
            if device.unique_id in device_ids:
                index = device_ids.index(device.unique_id)
                ul.create_daq_device(index, device)
                boards[index].board_num = index
                boards[index].initialize_ports()
                num_initialized += 1

    if num_initialized != num_boards:
        log.error(f'Error: {num_boards - num_initialized} boards not initialized. '
                  f'Check if the unique IDs exist and are plugged in. '
                  "Use 'get_detected_devices()' to see the available devices.")
        raise Exception(f'Error: {num_boards - num_initialized} boards not initialized. '
                        f'Check if the unique IDs exist and are plugged in. '
                        "Use 'get_detected_devices()' to see the available devices.")



class USB_3114(Board):
    '''Class representing the USB-3114 board.'''
    def __init__(self, name: str, unique_id: str, *pins: list[daq.PIN],
                 ao_range: enums.ULRange = enums.ULRange.UNI10VOLTS):
        super().__init__(name, unique_id, *pins)
        self.ao_range = ao_range

    def initialize_ports(self):
        self.getBoardInfo()

        # TODO: Implement initialization of digital ports (if needed for this device)
        return

class USB_1208FS_PLUS(Board):
    '''Class representing the USB-1208FS-Plus board.'''
    def __init__(self, name: str, unique_id: str, *pins: list[daq.PIN],
                 port_A_direction: enums.DigitalIODirection = enums.DigitalIODirection.OUT,
                 port_B_direction: enums.DigitalIODirection = enums.DigitalIODirection.OUT,
                 analog_input_mode: enums.AnalogInputMode = enums.AnalogInputMode.SINGLE_ENDED,
                 ai_range: enums.ULRange = enums.ULRange.UNI5VOLTS):
        # Initialize the board with the specified name and pins
        super().__init__(name, unique_id, *pins)

        self.port_A_direction = port_A_direction
        self.port_B_direction = port_B_direction
        self.analog_input_mode = analog_input_mode

        self.portA = 0b0
        self.portB = 0b0
        
        self.ai_range = ai_range
    
    def initialize_ports(self):
        self.getBoardInfo()

        # Set the digital port directions for the board
        enums.DigitalPortType.FIRSTPORTA
        ul.d_config_port(self.board_num, enums.DigitalPortType.FIRSTPORTA, self.port_A_direction)
        ul.d_config_port(self.board_num, enums.DigitalPortType.FIRSTPORTB, self.port_B_direction)

        # Set the analog input mode for the board
        ul.a_input_mode(self.board_num, self.analog_input_mode)
        return

class DIOPIN(daq.PIN):
    direction = daq.PIN_DIRECTION.IO
    type = daq.PIN_TYPE.DIGITAL
    supported_boards = [USB_3114]

    def __init__(self, net_name: str, channel: str):
        super().__init__(net_name, channel)
        
        raise NotImplementedError("DIOPIN (bit addressable) is not implemented yet for USB_3114. ")

class DIO_BUS_PIN(daq.PIN):
    direction = daq.PIN_DIRECTION.IO
    type = daq.PIN_TYPE.DIGITAL
    supported_boards = [USB_1208FS_PLUS]

    def __init__(self, net_name: str, channel: str):
        super().__init__(net_name, channel)
        
        if "A" in channel.upper():
            self.port = enums.DigitalPortType.FIRSTPORTA
        elif "B" in channel.upper():
            self.port = enums.DigitalPortType.FIRSTPORTB

    def dout(self, value:bool):
        '''
        Sets the digital value of the specified pin
        Parameters:
        pinName: str: The name of the pin to set the digital value of
        value: bool: The digital value to set the pin to
        '''

        super().dout(value)

        port = self.chnl[0].upper()  # Get the port letter (A or B)
        if port not in ['A', 'B']:
            log.error(f"Invalid port {port} for pin {self.net_name}."
                      f"Channel ({self.chnl}) must be from port 'A' or 'B'.")
            raise ValueError(f"Invalid port {port} for pin {self.net_name}."
                             f"Channel ({self.chnl}) must be from port 'A' or 'B'.")
        
        self.board: USB_1208FS_PLUS
        if port == 'A':
            port = enums.DigitalPortType.FIRSTPORTA
            port_state = self.board.portA
        elif port == 'B':
            port = enums.DigitalPortType.FIRSTPORTB
            port_state = self.board.portB

        channel = int(self.chnl[1:])  # Get the channel number as an integer

        x = port_state  # Get the current state of the port

        # set the bit to 0
        x &= ~(1 << channel)
        # set the bit to the value
        x = (x & ~(1 << channel)) | ((value & 1) << channel)

        if port == enums.DigitalPortType.FIRSTPORTA:
            self.board.portA = x
        elif port == enums.DigitalPortType.FIRSTPORTB:
            self.board.portB = x

        log.debug(f"Set pin {self.net_name} to {value} by writing "
                  f"{format(x, "08b")} to port {self.chnl[0].upper()}")

        log.debug(f"Writing digital output for pin {self.net_name} "
                  f"on port {port} with value {value} ")
        ul.d_out(self.board.board_num,
                 port,
                 value
                 )
        
    def reset(self):
        '''
        Resets the digital output pin to its default state (0).
        '''
        log.debug(f"Resetting digital output for pin {self.net_name} to 0")
        self.dout(False)

class AOPIN(daq.PIN):
    direction = daq.PIN_DIRECTION.OUTPUT
    type = daq.PIN_TYPE.ANALOG
    supported_boards = [USB_3114, USB_1208FS_PLUS]

    def vout(self, voltage: float):
        '''
        Sets the analog output voltage of the specified pin
        Parameters:
        pinName: str: The name of the pin to set the analog output voltage of
        voltage: float: The voltage to set the pin to
        '''

        super().vout(voltage)

        self.board: Board
        if self.board.ao_range is None:
            log.error(f"Analog output range not set for board {self.board.board_name}.")
            raise ValueError(f"Analog output range not set for board {self.board.board_name}.")
        
        log.debug(f"Setting analog output voltage for pin {self.net_name} "
                  f"to {voltage}V on channel {self.chnl} with range {self.board.ao_range}")
        ul.v_out(self.board.board_num, self.chnl, self.board.ao_range, voltage)

    def reset(self):
        '''
        Resets the analog output pin to its default state (0V).
        '''
        log.debug(f"Resetting analog output for pin {self.net_name} to 0V")
        self.vout(0.0)
        

class AIPIN(daq.PIN):
    direction = daq.PIN_DIRECTION.INPUT
    type = daq.PIN_TYPE.ANALOG
    supported_boards = [USB_1208FS_PLUS]

    def vin(self):
        super().vin()

        self.board: Board
        if self.board.ai_range is None:
            log.error(f"Analog input range not set for board {self.board.board_name}.")
            raise ValueError(f"Analog input range not set for board {self.board.board_name}.")

        log.debug(f"Reading analog input voltage for pin {self.net_name} "
                  f"on channel {self.chnl} with range {self.board.ai_range}")
        return ul.v_in(self.board.board_num, self.chnl, self.board.ai_range)
    

    def reset(self):
        return