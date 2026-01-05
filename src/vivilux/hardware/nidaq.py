'''Submodule providing interface to National Instruments DAQ hardware.
'''    

from time import time
from collections import namedtuple
from contextlib import contextmanager

import nidaqmx
import nidaqmx.system
from nidaqmx.constants import *

import numpy as np

from vivilux.logger import log
import vivilux.hardware.daq as daq

class Board(daq.Board):
    '''Base class for National Instruments DAQ boards.
    '''
    def __init__(self,
                 name: str,
                 dev_serial_num: int,
                 *pins: list[daq.PIN],
                 min_val: float = -0.0,
                 max_val: float = 2.0,
                 num_samples: int = 100,
                 num_samples_to_skip: int = 10,
                 terminal_config=TerminalConfiguration.RSE,
                 acquisition_type=AcquisitionType.FINITE,
                 ):
        super().__init__(name, *pins)
        self.unique_id = dev_serial_num
        self.min_val = min_val
        self.max_val = max_val
        self.num_samples = num_samples
        self.num_samples_to_skip = num_samples_to_skip
        self.terminal_config = terminal_config
        self.acquisition_type = acquisition_type
        
        self.pin_list: list[daq.PIN] = list(pins)
        self.pin_index_map: dict[str, int] = {}
        self.task = None

        self.lsb = (self.max_val - self.min_val) / (2**16)  # 16-bit ADC resolution
        
    def group_vin(self, pin_names, std=False):
        '''Groups the voltage inputs from the specified pins into a single array.
        
        Parameters
        ----------
        pin_names : list[str]
            A list of pin names to read the voltage inputs from.

        Returns
        -------
        np.ndarray
            An array containing the voltage inputs from the specified pins.
        '''
        if self.task is None:
            self.task: nidaqmx.Task = nidaqmx.Task()
            for idx, pin in enumerate(self.pin_list):
                if not isinstance(pin, AIPIN):
                    raise TypeError(f"Pin {pin.net_name} is not an AIPIN.")
                self.task.ai_channels.add_ai_voltage_chan(f"{self.board_num}/ai{pin.chnl}",
                                                            min_val=self.min_val,
                                                            max_val=self.max_val,
                                                            terminal_config=self.terminal_config)
                # Map pin name to its index in the task
                self.pin_index_map[pin.net_name] = idx
            self.task.timing.cfg_samp_clk_timing(rate=0.8*250e3/ len(self.pin_list), # 250 kHz divided by number of chann
                                                 sample_mode=self.acquisition_type,  # Explicitly set finite acquisition
                                                 samps_per_chan=self.num_samples
                                                )  
            self.task.start()
            self.task.read(number_of_samples_per_channel=self.num_samples)  # Initial read to set up the task
            self.task.stop() # Ensure task is stopped before reading
        
        self.task.start()
        full_data = np.array(self.task.read(number_of_samples_per_channel=self.num_samples))
        indices = [self.pin_index_map[pin_name] for pin_name in pin_names]
        raw_data = full_data[indices, self.num_samples_to_skip:]  # Select only the requested pins
        data = np.mean(raw_data, axis=1) # skips first few samples to avoid noise during a transition
        self.task.stop()
        
        if std:
            return data, np.std(raw_data, axis=1)
        return data
    
    def group_scan_vin(self, pin_names, num_samples = 100):
        '''Groups the voltage inputs from the specified pins into a single array
            by scanning multiple samples.

        Parameters
        ----------
        pin_names : list[str]
            A list of pin names to read the voltage inputs from.
        num_samples : int
            The number of samples to read from each pin.

        Returns
        -------
        np.ndarray
            An array containing the voltage inputs from the specified pins.
        '''
        with nidaqmx.Task() as task:
            for pin_name in pin_names:
                pin = self.pins[pin_name]
                if not isinstance(pin, AIPIN):
                    raise TypeError(f"Pin {pin_name} is not an AIPIN.")
                task.ai_channels.add_ai_voltage_chan(f"{self.board_num}/ai{pin.chnl}",
                                                     min_val=self.min_val,
                                                     max_val=self.max_val,
                                                     terminal_config=self.terminal_config)
            data = np.array(task.read(number_of_samples_per_channel=self.num_samples))
        return data
    
    @contextmanager
    def synchronized_acquisition(self, 
                                  pin_names: list[str], 
                                  duration_ms: float = 10.0,
                                  rate: float = 250e3):
        '''Context manager for synchronized finite acquisition.
        
        Creates and starts the acquisition task, then yields control so the user
        can trigger their modulation event. When exiting, waits for the acquisition
        to complete and stores the data.
        
        Parameters
        ----------
        pin_names : list[str]
            List of analog input pin names to acquire from
        duration_ms : float
            Duration to acquire data in milliseconds (default: 10.0)
        rate : float
            Sample rate in Hz (default: 250e3)
            
        Yields
        ------
        None
            Control returns after task is started and acquiring
        '''
        num_samples = int(duration_ms * 1e-3 * rate)
        
        task = nidaqmx.Task()
        try:
            # Add channels
            for pin_name in pin_names:
                pin = self.pins[pin_name]
                if not isinstance(pin, AIPIN):
                    raise TypeError(f"Pin {pin_name} is not an AIPIN.")
                task.ai_channels.add_ai_voltage_chan(
                    f"{self.board_num}/ai{pin.chnl}",
                    min_val=self.min_val,
                    max_val=self.max_val,
                    terminal_config=self.terminal_config
                )
            
            # Configure finite acquisition
            task.timing.cfg_samp_clk_timing(
                rate=rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=num_samples
            )
            
            # Start acquisition
            task.start()
            start_time = time()
            log.info(f"Started synchronized acquisition: {num_samples} samples @ {rate/1e3:.1f} kHz")
            
            # Yield control - user can now modulate their parameter
            # Task is running in hardware, acquiring data
            yield
            
            # Wait for acquisition to complete
            task.wait_until_done(timeout=duration_ms * 1e-3 + 1.0)
            
            # Read all data
            data = np.array(task.read(number_of_samples_per_channel=num_samples))
            
            # Generate timestamps
            timestamps = np.arange(num_samples) / rate
            
            # Store for retrieval
            self._last_acquisition = {
                "start_time": start_time,
                "timestamps": timestamps,
                "data": data
            }
            
            log.info("Synchronized acquisition complete")
        
        except Exception as e:
            err_msg = f"Error during synchronized acquisition: {e}"
            log.error(err_msg)
            raise RuntimeError(err_msg)
            
        finally:
            task.close()

    def get_acquisition_data(self) -> tuple[np.ndarray, np.ndarray]:
        '''Get the most recent synchronized acquisition data.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (timestamps, data) from the last acquisition
        '''
        if not hasattr(self, '_last_acquisition'):
            err_msg = "No acquisition data available. Run synchronized_acquisition first."
            log.error(err_msg)
            raise RuntimeError(err_msg)
        return self._last_acquisition
    
    def __del__(self):
        '''Cleanup: close the task when the board object is destroyed.'''
        if self.task is not None:
            try:
                self.task.close()
                log.info("NI DAQ task closed during cleanup.")
            except:
                log.error("Error closing NI DAQ task during cleanup.")

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
            device.reset_device()
            if device.dev_serial_num in device_ids:
                index = device_ids.index(device.dev_serial_num)
                boards[index].board_num = device.name
                num_initialized += 1
    else:
        log.info(f'Found {len(system.devices)} NI DAQ device(s):')
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
    def __init__(self, name: str, dev_serial_num: int, *pins: list[daq.PIN], **kwargs):
        super().__init__(name, dev_serial_num, *pins, **kwargs)

# TODO: implement std deviation option for vin like group_vin above
class AIPIN(daq.PIN):
    direction = daq.PIN_DIRECTION.INPUT
    type = daq.PIN_TYPE.ANALOG
    supported_boards = [USB_6210]

    def vin(self):
        super().vin()

        self.board: Board
        min_val = self.board.min_val
        max_val = self.board.max_val
        terminal_config = self.board.terminal_config

        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{self.board.board_num}/ai{self.chnl}",
                                                 min_val=min_val,
                                                 max_val=max_val,
                                                 terminal_config=terminal_config)
            task.timing.cfg_samp_clk_timing(rate=250e3)
            data = np.array(task.read(number_of_samples_per_channel=self.board.num_samples))
            data = np.mean(data[self.board.num_samples_to_skip:]) # skips first few samples to avoid noise from initialization
            # TODO: make these values configurable for different sampling and averaging
        
        # log.debug(f"Read 100 samples from {self.board.board_num}/ai{self.chnl}"
        #           " (skipped first 10 samples and returned mean)")
        return data
    
    def scan_vin(self, num_samples = 100, rate=250e3):
        super().scan_vin(num_samples)

        self.board: Board
        min_val = self.board.min_val
        max_val = self.board.max_val
        terminal_config = self.board.terminal_config
        
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{self.board.board_num}/ai{self.chnl}",
                                                 min_val=min_val,
                                                 max_val=max_val,
                                                 terminal_config=terminal_config)
            task.timing.cfg_samp_clk_timing(rate=rate,)
            data = np.array(task.read(number_of_samples_per_channel=num_samples))
        
        # log.debug(f"Read {num_samples} samples from {self.board.board_num}/ai{self.chnl}")
        return data

    def reset(self):
        return