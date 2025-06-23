'''Submodule defining abstractions for common elements of data acquisition
    boards and a wrapper for more easily handling net names in a single
    object containing all the pin definitions.
'''

from math import nan

import jax
import jax.numpy as jnp
import numpy as np

from vivilux.logger import log

class PIN_DIRECTION:
    INPUT = 'I'
    OUTPUT = 'O'
    IO = 'IO'  # Input/Output, used for digital pins

class PIN_TYPE:
    ANALOG = 'A'  # Analog pin, used for analog pins
    DIGITAL = 'D'  # Digital pin, used for digital pins

class PIN:
    direction = None  # To be defined in subclasses
    type = None  # To be defined in subclasses
    supported_boards: list[str] = []  # List of supported boards for this pin class

    def __init__(self, net_name: str, channel: str):
        self.net_name = net_name  # Name of the pin
        self.chnl = channel  # Channel name
        self.board = None

    def assign_board(self, board: "Board"):
        '''Assigns a board to this pin.
        
        Parameters
        ----------
        board : Board
            The board to assign to this pin.
        '''
        if board.__class__ not in self.supported_boards:
            log.error(f"Pin {self.net_name} is not supported on board {board.board_name}")
            raise ValueError(f"Pin {self.net_name} is not supported on board {board.board_name}")
        self.board = board

    def reset(self):
        '''Resets the pin to its default state.
        
        For analog pins, this sets the voltage to 0V.
        For digital pins, this sets the value to low (False).
        '''
        log.error(f"Reset method not implemented for pin {self.net_name} of type {self.type}")
        raise NotImplementedError(f"Reset method not implemented for pin {self.net_name} of type {self.type}")

    def vout(self, voltage: float):
        '''Sets the voltage of the specified pin.
        
        Parameters
        ----------
        pin_name : str
            The name of the pin to set the voltage of.
        voltage : float
            The voltage to set the pin to.
        '''
        if self.board is None:
            log.error(f"Tried to set {self.net_name} when board is not assigned.")
            raise RuntimeError(f"Board is not assigned for {self.net_name}." \
                               " Please assign a board number before using this method.")
        if self.direction != PIN_DIRECTION.OUTPUT:
            raise ValueError(f"Pin {self.net_name} is not an output pin")
        if self.type != PIN_TYPE.ANALOG:
            raise ValueError(f"Pin {self.net_name} is not an analog pin")
    
        log.debug(f"(ABSTRACT PIN) Setting pin {self.net_name} to {voltage}V on board {self.board.board_name}")
        return
    
    def vin(self) -> float:
        '''Reads the voltage from the analog input pin.
        
        Returns
        -------
        float
            The voltage read from the pin.
        '''
        if self.board is None:
            log.error(f"Tried to read {self.net_name} when board is not assigned.")
            raise RuntimeError(f"Board is not assigned for {self.net_name}." \
                               " Please assign a board number before using this method.")
        
        if self.direction != PIN_DIRECTION.INPUT:
            log.error(f"Pin {self.net_name} is not an input pin")
            raise ValueError(f"Pin {self.net_name} is not an input pin")
        if self.type != PIN_TYPE.ANALOG:
            log.error(f"Pin {self.net_name} is not an analog pin")
            raise ValueError(f"Pin {self.net_name} is not an analog pin")

        log.debug(f"(ABSTRACT PIN) Reading pin {self.net_name} on board {self.board.board_name}")        
        return nan  # Placeholder for actual implementation
    
    def scan_vin(self, num_samples: int = 100) -> list[float]:
        '''Reads multiple samples from the analog input pin.
        
        Parameters
        ----------
        num_samples : int, optional
            The number of samples to read from the pin (default is 100).
        
        Returns
        -------
        list[float]
            A list of voltages read from the pin.
        '''
        if self.board is None:
            log.error(f"Tried to scan {self.net_name} when board is not assigned.")
            raise RuntimeError(f"Board is not assigned for {self.net_name}." \
                               " Please assign a board number before using this method.")
        
        if self.direction != PIN_DIRECTION.INPUT:
            log.error(f"Pin {self.net_name} is not an input pin")
            raise ValueError(f"Pin {self.net_name} is not an input pin")
        if self.type != PIN_TYPE.ANALOG:
            log.error(f"Pin {self.net_name} is not an analog pin")
            raise ValueError(f"Pin {self.net_name} is not an analog pin")
        
        log.debug(f"(ABSTRACT PIN) Scanning {num_samples} samples from pin {self.net_name} on board {self.board.board_name}")
        return [nan] * num_samples # Placeholder for actual implementation
    
    def dout(self, value: bool):
        '''Sets the digital value of the specified pin.
        
        Parameters
        ----------
        value : bool
            The digital value to set the pin to.
        '''
        if self.board is None:
            log.error(f"Tried to set {self.net_name} when board is not assigned.")
            raise RuntimeError(f"Board is not assigned for {self.net_name}." \
                               " Please assign a board number before using this method.")
        
        if self.direction != PIN_DIRECTION.OUTPUT or self.direction != PIN_DIRECTION.IO:
            log.error(f"Pin {self.net_name} is not an output pin")
            raise ValueError(f"Pin {self.net_name} is not an output pin")
        if self.type != PIN_TYPE.DIGITAL:
            log.error(f"Pin {self.net_name} is not a digital pin")
            raise ValueError(f"Pin {self.net_name} is not a digital pin")
        
        log.debug(f"(ABSTRACT PIN) Setting pin {self.net_name} to {value} on board {self.board.board_name}")
        return


class Board:
    '''Base class for all boards. Provides methods to read and write to pins
        after the board has assigned a board number.
    '''
    def __init__(self, name: str, *pins: list[PIN]):
        self.board_num = None
        self.board_name = name
        
        self.pins: dict[str, PIN] = {}
        self.nets: list[str] = []

        for pin in pins:
            if not isinstance(pin, PIN):
                raise TypeError(f"Expected PIN instance, got {type(pin)}")
            pin.assign_board(self)
            # check for duplicate pin names
            if pin.net_name in self.pins:
                log.error(f"Duplicate pin name {pin.net_name} found in board {self.board_name}")
                raise ValueError(f"Duplicate pin name {pin.net_name} found in board {self.board_name}")
            self.pins[pin.net_name] = pin
            # check for duplicate net names
            if pin.net_name in self.nets:
                log.error(f"Duplicate net name {pin.net_name} found in board {self.board_name}")
                raise ValueError(f"Duplicate net name {pin.net_name} found in board {self.board_name}")
            self.nets.append(pin.net_name)

    def assign_board_num(self, board_num: int):
        '''Assigns a board number to this board.
        Parameters
        ----------
        board_num : int
            The board number to assign to this board.
        Raises
        ------
        RuntimeError
            If the board number is already assigned or if the board number is None.
        '''
        if self.board_num is not None:
            log.error(f"Board number is already assigned for {self.board_name}")
            raise RuntimeError(f"Board number is already assigned for {self.board_name}.")
        if not isinstance(board_num, int):
            log.error(f"Board number must be an integer, got {type(board_num)}")
            raise TypeError(f"Board number must be an integer, got {type(board_num)}")
        if board_num < 0:
            log.error(f"Board number must be a non-negative integer, got {board_num}")
            raise ValueError(f"Board number must be a non-negative integer, got {board_num}")
        self.board_num = board_num
        log.info(f"Assigned board number {self.board_num} to board {self.board_name}")
            

    def vout(self, pin_name: str, voltage: float):
        '''Sets the voltage of the specified pin.
        
        Parameters
        ----------
        pin_name : str
            The name of the pin to set the voltage of.
        voltage : float
            The voltage to set the pin to.
        '''
        if self.board_num is None:
            log.error(f"Tried to set {pin_name} when board number is not set")
            raise RuntimeError(f"Board number is not set for {self.board_name}." \
                               " Please assign a board number before using this method.")

        if pin_name not in self.pins:
            raise ValueError(f"Pin {pin_name} not found in board {self.board_name}")
        pin = self.pins[pin_name]
        if pin.direction != PIN_DIRECTION.OUTPUT:
            raise ValueError(f"Pin {pin_name} is not an output pin")
        if pin.type != PIN_TYPE.ANALOG:
            raise ValueError(f"Pin {pin_name} is not an analog pin")

        # log.debug(f"Set pin {pinName} to {voltage}V")
        # ul.v_out(self.board_num, pin.chnl, ao_range, voltage)
        log.debug(f"(ABSTRACT PIN) Setting pin {pin_name} to {voltage}V on board {self.board_name}")
        self.pins[pin_name].vout(voltage)
    
        return

    def vin(self, pin_name: str) -> float:
        '''Reads the voltage from the specified pin.
        
        Parameters
        ----------
        pin_name : str
            The name of the pin to read from.
        
        Returns
        -------
        float
            The voltage read from the pin.
        '''
        if self.board_num is None:
            log.error(f"Tried to set {pin_name} when board number is not set")
            raise RuntimeError(f"Board number is not set for {self.board_name}." \
                               " Please assign a board number before using this method.")

        if pin_name not in self.pins:
            raise ValueError(f"Pin {pin_name} not found in board {self.board_name}")
        pin = self.pins[pin_name]

        log.debug(f"(ABSTRACT PIN) Reading pin {pin_name} on board {self.board_name}")
        return pin.vin()

    def scan_vin(self, pin_name: str, num_samples: int = 100) -> list[float]:
        '''Reads multiple samples from the specified pin.
        
        Parameters
        ----------
        pin_name : str
            The name of the pin to read from.
        num_samples : int, optional
            The number of samples to read from the pin (default is 100).
        
        Returns
        -------
        list[float]
            A list of voltages read from the pin.
        '''
        if self.board_num is None:
            log.error(f"Tried to scan {pin_name} when board number is not set")
            raise RuntimeError(f"Board number is not set for {self.board_name}." \
                               " Please assign a board number before using this method.")

        if pin_name not in self.pins:
            raise ValueError(f"Pin {pin_name} not found in board {self.board_name}")
        pin = self.pins[pin_name]

        log.debug(f"(ABSTRACT PIN) Scanning {num_samples} samples from pin"
                  f" {pin_name} on board {self.board_name}")
        return pin.scan_vin(num_samples)

    def group_vin(self, pin_names: list[str]) -> jnp.ndarray:
        '''Reads multiple samples from the specified pins.
        
        Parameters
        ----------
        pin_names : list[str]
            A list of pin names to read from.
        
        Returns
        -------
        jnp.ndarray
            An array of voltages read from the pins (mock implementation).
        '''
        if self.board_num is None:
            log.error(f"Tried to group scan when board number is not set")
            raise RuntimeError(f"Board number is not set for {self.board_name}." \
                               " Please assign a board number before using this method.")

        for pin_name in pin_names:
            if pin_name not in self.pins:
                raise ValueError(f"Pin {pin_name} not found in board {self.board_name}")
            
        log.debug(f"(ABSTRACT PIN) Group scanning pins {pin_names} on board {self.board_name}")
        return jnp.full((len(pin_names),), nan)  # Placeholder for actual implementation
    
    def group_scan_vin(self, pin_names: list[str], num_samples: int = 100) -> jnp.ndarray:
        '''Reads multiple samples from the specified pins.
        
        Parameters
        ----------
        pin_names : list[str]
            A list of pin names to read from.
        num_samples : int, optional
            The number of samples to read from each pin (default is 100).
        
        Returns
        -------
        jnp.ndarray
            A 2D array where each row corresponds to a pin and each column corresponds to a sample (mock implementation).
        '''
        if self.board_num is None:
            log.error(f"Tried to group scan when board number is not set")
            raise RuntimeError(f"Board number is not set for {self.board_name}." \
                               " Please assign a board number before using this method.")

        for pin_name in pin_names:
            if pin_name not in self.pins:
                raise ValueError(f"Pin {pin_name} not found in board {self.board_name}")
        
        log.debug(f"(ABSTRACT PIN) Group scanning {num_samples} samples from "
                  f"pins {pin_names} on board {self.board_name}")
        return jnp.full((num_samples, len(pin_names)), nan)
        
    
    def dout(self, pin_name: str, value: bool):
        '''Sets the digital value of the specified pin.
        
        Parameters
        ----------
        pin_name : str
            The name of the pin to set the digital value of.
        value : bool
            The digital value to set the pin to.
        '''
        if self.board_num is None:
            log.error(f"Tried to set {pin_name} when board number is not set")
            raise RuntimeError(f"Board number is not set for {self.board_name}." \
                               " Please assign a board number before using this method.")
        
        if pin_name not in self.pins:
            raise ValueError(f"Pin {pin_name} not found in board {self.board_name}")
        pin = self.pins[pin_name]
        if pin.direction != PIN_DIRECTION.OUTPUT:
            raise ValueError(f"Pin {pin_name} is not an output pin")
        if not pin.type == PIN_TYPE.DIGITAL:
            raise ValueError(f"Pin {pin_name} is not a digital pin")

        self.pins[pin_name].dout(value)
    
    def reset_board(self):
        '''Resets the board by setting all output pins to 0V and digital pins to low.
        '''
        if self.board_num is None:
            log.error(f"Tried to reset board {self.board_name} when board number is not set")
            raise RuntimeError(f"Board number is not set for {self.board_name}."\
                               " Please assign a board number before using this method.")
        
        for pin_name, pin in self.pins.items():
            if pin.direction == PIN_DIRECTION.OUTPUT:
                pin.reset()
            elif pin.direction == PIN_DIRECTION.IO:
                pin.reset()
        
        return

class Netlist:
    '''A class which contains all the net names and provides access to their
        corresponding pins for easier control over the experiment using the net
        name instead having the user keep track of the board and port names.

        example usage:
        netlist = Netlist(board1, board2)
        netlist["net_name"].vout(5.0)  # Sets the voltage of the pin with net name "net_name" to 5.0V
        netlist["net_name"].vin()  # Reads the voltage from the pin with net name "net_name"
    '''
    def __init__(self, *boardlist: list["Board"]):
        '''Initializes the Netlist with a list of boards and associates
            their net names with the corresponding pins.
        '''
        self.boardlist = boardlist
        self.board_dict: dict[str, Board] = {} # board name to board mapping
        self.pins_dict: dict[str, PIN] = {} # net name to pin mapping
        self.nets: list[str] = []

        self.in_context = False  # Flag to indicate if the netlist is being used in a context manager

        for board in boardlist:
            if not isinstance(board, Board):
                log.error(f"Expected Board instance, got {type(board)}")
                raise TypeError(f"Expected Board instance, got {type(board)}")
            if board.board_name in self.board_dict:
                log.error(f"Duplicate board name {board.board_name} found in netlist")
                raise ValueError(f"Duplicate board name {board.board_name} found in netlist")
            self.board_dict[board.board_name] = board
            for net in board.nets:
                if net in self.nets:
                    log.error(f"Duplicate net name {net} found in netlist. " \
                                f"Already found in board {self.pins_dict[net].board.board_name}")
                    raise ValueError(f"Duplicate net name {net} found in netlist. " \
                                        f"Already found in board {self.pins_dict[net].board.board_name}")
                self.nets.append(net)
                self.pins_dict[net] = board.pins[net]

    # get pin by net name using the __getitem__ method
    def __getitem__(self, net_name: str) -> PIN:
        '''Gets the pin corresponding to the given net name.
        
        Parameters
        ----------
        net_name : str
            The net name to get the pin for.
        
        Returns
        -------
        PIN
            The pin corresponding to the given net name.
        '''
        if not self.in_context:
            log.error("Netlist is being accessed (getitem) outside context manager. ")
            raise RuntimeError("Netlist is being accessed (getitem) outside context manager. " \
                               "Please use the Netlist in a context manager.")
        
        if net_name not in self.pins_dict:
            log.error(f"Net name {net_name} not found in netlist")
            raise KeyError(f"Net name {net_name} not found in netlist")
        return self.pins_dict[net_name]

    def __enter__(self):
        '''Context manager setup for the netlist.
        '''
        # TODO: implement setup code (if any)
        self.in_context = True
        log.debug(f"(NETLIST) Entering context manager for netlist with boards:"
                  f" {list(self.board_dict.keys())}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''Resets all boards in the netlist when exiting the context manager.
        '''
        for board in self.board_dict.values():
            board.reset_board()
        self.in_context = False
        log.debug(f"(NETLIST) Exiting context manager for netlist with boards:"
                  f" {list(self.board_dict.keys())}")
        # Returning False propagates exceptions, True suppresses them
        return False

    def group_vin(self, net_names: list[str]) -> np.ndarray:
        '''Reads multiple samples from the specified net names.
        
        Parameters
        ----------
        net_names : list[str]
            A list of net names to read from.
        
        Returns
        -------
        np.ndarray
            A 2D array where each row corresponds to a net name and each column
            corresponds to a sample.
        '''

        if not self.in_context:
            log.error("Netlist is being accessed (group_vin) outside context manager.")
            raise RuntimeError("Netlist is being accessed (group_vin) outside context manager. " \
                               "Please use the Netlist in a context manager.")

        if not isinstance(net_names, list):
            log.error(f"Expected list of net names, got {type(net_names)}")
            raise TypeError(f"Expected list of net names, got {type(net_names)}")
        
        board_name = None
        for net_name in net_names:
            if net_name not in self.pins_dict:
                log.error(f"Net name {net_name} not found in netlist")
                raise KeyError(f"Net name {net_name} not found in netlist")
            
            if board_name is None:
                board_name = self.pins_dict[net_name].board.board_name
            else:
                if self.pins_dict[net_name].board.board_name != board_name:
                    log.error(f"Net names {net_names} belong to different boards and cannot be read as group: "
                              f"{self.pins_dict[net_name].board.board_name} and {board_name}")
                    raise ValueError(f"Net names {net_names} belong to different boards and cannot be read as group: "
                                     f"{self.pins_dict[net_name].board.board_name} and {board_name}")
        
        log.debug(f"(NETLIST) Group scanning pins {net_names}")
        return self.board_dict[board_name].group_vin(net_names)
    
    def group_scan_vin(self, net_names: list[str], num_samples: int = 100) -> np.ndarray:
        '''Reads multiple samples from the specified net names.
        
        Parameters
        ----------
        net_names : list[str]
            A list of net names to read from.
        num_samples : int, optional
            The number of samples to read from each pin (default is 100).
        
        Returns
        -------
        np.ndarray
            A 2D array where each row corresponds to a net name and each column
            corresponds to a sample.
        '''
        if not self.in_context:
            log.error("Netlist is being accessed (group_scan_vin) outside context manager.")
            raise RuntimeError("Netlist is being accessed (group_scan_vin) outside context manager. " \
                               "Please use the Netlist in a context manager.")

        if not isinstance(net_names, list):
            log.error(f"Expected list of net names, got {type(net_names)}")
            raise TypeError(f"Expected list of net names, got {type(net_names)}")
        
        board_name = None
        for net_name in net_names:
            if net_name not in self.pins_dict:
                log.error(f"Net name {net_name} not found in netlist")
                raise KeyError(f"Net name {net_name} not found in netlist")
            
            if board_name is None:
                board_name = self.pins_dict[net_name].board.board_name
            else:
                if self.pins_dict[net_name].board.board_name != board_name:
                    log.error(f"Net names {net_names} belong to different boards and cannot be read as group: "
                              f"{self.pins_dict[net_name].board.board_name} and {board_name}")
                    raise ValueError(f"Net names {net_names} belong to different boards and cannot be read as group: "
                                     f"{self.pins_dict[net_name].board.board_name} and {board_name}")
        
        log.debug(f"(NETLIST) Group scanning {num_samples} samples from pins {net_names}")
        return self.board_dict[board_name].group_scan_vin(net_names, num_samples)