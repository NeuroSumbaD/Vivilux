'''Submodule providing interface to National Instruments DAQ hardware.
'''    

from collections import namedtuple

import nidaqmx
import nidaqmx.system
import nidaqmx.constants

import numpy as np

from vivilux.logger import log
import vivilux.hardware.daq as daq

class Board(daq.Board):
    '''Base class for National Instruments DAQ boards.
    '''
    def __init__(self, name: str, dev_serial_num: int, *pins: list[daq.PIN]):
        super().__init__(name, *pins)
        self.unique_id = dev_serial_num


def get_driver_version() -> namedtuple:
    system = nidaqmx.system.System.local()
    print("Driver version: ", system.driver_version)
    return system.driver_version

def get_devices():
    system = nidaqmx.system.System.local()
    print(f"Found {len(system.local().devices)} devices:")
    for device in system.devices:
        print("Device:\n", device) 


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

    system = nidaqmx.system.System.local()
    
    if len(system.devices) == 0:
        raise Exception('Error: No NI DAQ devices found.')
    
    device_ids = [board.dev_serial_num for board in system.devices]

    num_boards = len(device_ids)
    num_initialized = 0
    
    if verbose:
        print('Found', len(system.devices), 'NI DAQ device(s):')
        for device in system.devices:
            print(f'Name: {device.product_type} ({device.dev_serial_num}) - '
                  f'Device ID = {device.product_num}') 
            if device.dev_serial_num in device_ids:
                index = device_ids.index(device.dev_serial_num)
                boards[index].board_num = device.name
                num_initialized += 1
    else:
        log.info('Found', len(system.devices), 'NI DAQ device(s):')
        for device in system.devices:
            log.info(f'Name: {device.product_type} ({device.dev_serial_num}) - '
                     f'Device ID = {device.product_num}') 
            if device.dev_serial_num in device_ids:
                index = device_ids.index(device.dev_serial_num)
                boards[index].board_num = device.name
                num_initialized += 1

    if num_initialized != num_boards:
        log.error(f'Error: {num_boards - num_initialized} boards not initialized. '
                  f'Check if the unique IDs exist and are plugged in. '
                  "Use 'get_detected_devices()' to see the available devices.")
        raise Exception(f'Error: {num_boards - num_initialized} boards not initialized. '
                        f'Check if the unique IDs exist and are plugged in. '
                        "Use 'get_detected_devices()' to see the available devices.")

class USB_6210(Board):
    '''NI USB-6210 DAQ board.
    '''
    def __init__(self, name: str, dev_serial_num: int, *pins: list[daq.PIN]):
        super().__init__(name, dev_serial_num, *pins)

class AIPIN(daq.PIN):
    direction = daq.PIN_DIRECTION.INPUT
    type = daq.PIN_TYPE.ANALOG
    supported_boards = [USB_6210]

    def vin(self):
        super().vin()

        self.board: Board
        
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{self.board.board_num}/ai{self.chnl}",
                                                 min_val=-0.0,
                                                 max_val=2.0,
                                                 terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            data = np.array(task.read(number_of_samples_per_channel=100))
            data = np.mean(data[10:],axis=1) # skips first few samples to avoid noise from initialization
            # TODO: make these values configurable for different sampling and averaging
        
        log.debug(f"Read 100 samples from {self.board.board_num}/ai{self.chnl}"
                  " (skipped first 10 samples and returned mean)")
        return data
    
    def scan_vin(self, num_samples = 100):
        super().scan_vin(num_samples)

        self.board: Board
        
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{self.board.board_num}/ai{self.chnl}",
                                                 min_val=-0.0,
                                                 max_val=2.0,
                                                 terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            data = np.array(task.read(number_of_samples_per_channel=num_samples))
        
        log.debug(f"Read {num_samples} samples from {self.board.board_num}/ai{self.chnl}")
        return data

    def reset(self):
        return