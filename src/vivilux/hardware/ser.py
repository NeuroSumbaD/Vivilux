'''Serial communication with a microcontroller over USB. The microcontroller is
    expected to adhere to the protocol defined in the pico_daq repository:
        https://github.com/ngncs-neuromorphic/pico2_daq

    The communication protocol is as follows:
    - Commands are sent as plain text strings, terminated with a newline character.
    - Responses from the microcontroller are also plain text strings, terminated with a newline character.
    - The microcontroller will acknowledge board definitions with "ACK" and the board info.
    - If a command is not recognized, the microcontroller will respond with "ERROR: <message>".
'''

import serial
import time
import serial.tools.list_ports

from vivilux.logger import log
from vivilux.hardware import daq

class Board(daq.Board):
    '''Base class for DAQ boards connected to the microcontroller defined by
        the 'BoardManager' class.

        Parameters
        ----------
        name : str
            The human-readable name of the board.
        uid : int
            A unique integer identifier for the board.
        csPin : int
            The chip select pin for the SPI connection to the board.
        pins : list[daq.PIN]
            The list of pin definitions on the board.
    '''
    type: str = None
    def __init__(self, name: str, uid: int, csPin: int,
                 *pins: list[daq.PIN],
                 **kwargs,
                 ):
        super().__init__(name,
                         *pins)
        self.uid = uid
        self.csPin = csPin
        self.ao_range = None if "ao_info" not in kwargs else kwargs["ao_info"]
        self.ai_range = None if "ai_info" not in kwargs else kwargs["ai_info"]

        self.board_manager: 'BoardManager' = None
        self.serial: serial.Serial = None

    def serial_vout(self, channel: int, voltage: float):
        '''Sends a command over serial to set the analog output voltage of the
            specified channel on this board.

            Parameters
            ----------
            channel : int
                The channel number on the board to set the voltage.
            voltage : float
                The voltage to set the channel to.
        '''
        if self.serial is None:
            err_msg = f"Error: Attempted to set voltage on board {self.board_name} " \
                       "before running 'ser.config_detected_devices()'."
            log.error(err_msg)
            raise RuntimeError(err_msg)

        command = f"W{self.uid},{channel},{voltage}\n"
        self.serial.write(command.encode())
        # log.info(f"Board {self.board_name} sent command: {command.strip()}")
        response = self.serial.readline().decode().strip()
        if "ACK,W" in response:
            # log.info(f"Board {self.board_name} acknowledged command with: {response}")
            '''No action needed'''
        elif "ERROR:" in response:
            log.error(f"Error occurred while setting voltage on "
                      f"{self.board_name} channel {channel}: {response}")
        else:
            log.warning(f"Received unexpected response: {response}")

class BoardManager(daq.Board):
    '''The BoardManager is a microcontroller board that manages multiple DAQ
        boards and handles the USB serial connection to the computer. Multiple
        DAQ boards will be connected to the GPIO pins of the microcontroller
        according to the microcontroller's interface and pinout.

        NOTE: The name of this board is defined by the user when flashing the
        microcontroller firmware, so the name of this board also serves as its
        unique identifier.
    '''
    def __init__(self, name: str,  *boards_list: list[Board]):
        '''Unlike most boards, the 'BoardManager' contains a list of boards
            rather than pins because the pins are assigned to the child board
            objects when they are created.
        '''
        self.board_name = name
        self.boards: dict[str, Board] = {}

        self.pins: dict[str, daq.PIN] = {}
        self.nets: list[str] = []

        self.serial: serial.Serial | None = None

        for board in boards_list:
            if not isinstance(board, Board):
                raise TypeError(f"Expected pydaq.serial.Board instance, got {type(board)}")
            # check for duplicate board names
            if board.board_name in self.boards:
                log.error(f"Duplicate board name {board.board_name} found in manager {self.board_name}")
                raise ValueError(f"Duplicate board name {board.board_name} found in manager {self.board_name}")
            self.boards[board.board_name] = board
            board.board_manager = self
            # check for duplicate net names
            for pin_name in board.pins:
                # check for duplicate pin names
                if pin_name in self.pins:
                    log.error(f"Duplicate pin name {pin_name} found in board {board.board_name}")
                    raise ValueError(f"Duplicate pin name {pin_name} found in board {board.board_name}")
                self.pins[pin_name] = board.pins[pin_name]
                # Check for duplicate net names
                if pin_name in self.nets:
                    log.error(f"Duplicate net name {pin_name} found in board {board.board_name}")
                    raise ValueError(f"Duplicate net name {pin_name} found in board {board.board_name}")
                self.nets.append(pin_name)

    def initialize_board(self):
        '''Sends serial commands to define the DAC and ADCs connected to the
            microcontroller.
        '''
        if self.serial is None:
            err_msg = f"Error: Attempted to initialize board {self.board_name} " \
                       "before running 'ser.config_detected_devices()'."
            log.error(err_msg)
            raise RuntimeError(err_msg)

        log.info(f"Initializing board definitions for {self.board_name}")

        log.info(f"Resetting initialized DAC/ADCs in parser")
        self.serial.write(b"RESET\n")
        response = self.serial.readline().decode().strip()
        if response != "ACK,RESET":
            log.error(f"Error: Reset failed with response: {response}")
            raise RuntimeError(f"Error: Reset failed with response: {response}")
        
        self.defineBoards()
        self.serial.write(b"ENDHS\n")
        response = self.serial.readline().decode().strip()
        if response != "HSOK":
            log.error(f"Error: Handshake failed with response: {response}")
            raise RuntimeError(f"Error: Handshake failed with response: {response}")
        else:
            log.info(f"HSOK: Handshake with {self.board_name} successful.")
        
        for board in self.boards.values():
            board.serial = self.serial

    def defineBoards(self,):
        '''Sends the BOARD definition commands for each child board to the
            microcontroller over the serial connection.
        '''
        for board in self.boards.values():
            self.defineBoard(board)

    def defineBoard(self, board: Board):
        '''Sends a BOARD definition command to the Pico.
        '''
        command = f"BOARD,{board.type},{board.board_name},{board.uid},{board.csPin}\n"
        self.serial.write(command.encode())
        log.info(f"Sent board command: {command.strip()}")
        response = self.serial.readline().decode().strip()
        if "ACK BOARD" in response:
            log.info(f"Board {self.board_name} acknowledged command with: {response}")
        elif "ERROR:" in response:
            log.error("Error occurred while defining board "
                      f"{board.board_name}: {response}")
            raise RuntimeError(f"Error occurred while defining board "
                               f"{board.board_name}: {response}")
        else:
            log.warning(f"Received unknown board definition response: {response}")
        return response

class EVAL_AD5370(Board):
    '''Class representing the EVAL_AD5370 40-channel DAC boards. Methods
        defined in this class will interact with any specific features
        unique to this board.
    '''
    type: str = "DAC"
    def __init__(self, name: str, uid: int, csPin: int,
                 *pins: list[daq.PIN],
                 ao_ranges: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (),):
        super().__init__(name,
                         uid,
                         csPin,
                         *pins)
        self.ao_ranges = ao_ranges
        
class VC_709(BoardManager):
    '''Class representing the VC-709 FPGA board. Methods
        defined in this class will interact with any specific features
        unique to this board.
    '''
    # TODO: Modify the base class to more clearly distinguish between
    # BoardManager and FPGA or other serial boards (maybe top-level 
    # SerialBoard class with and initialization code)
    def __init__(self, name: str,
                 *pins: list[daq.PIN],
                 **kwargs,
                 ):
        '''
        Parameters
        ----------
        name : str
            The UID programmed into the board manager for COM port identification.
        pins : list[daq.PIN]
            The list of pin definitions on the board.
        '''
        self.board_name = name
        
        self.pins: dict[str, daq.PIN] = {}
        self.nets: list[str] = []

        for pin in pins:
            if not isinstance(pin, daq.PIN):
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
        
    def initialize_board(self):
        '''Sends serial commands to put the FPGA laser into a known state.
        '''
        if self.serial is None:
            err_msg = f"Error: Attempted to initialize board {self.board_name} " \
                       "before running 'ser.config_detected_devices()'."
            log.error(err_msg)
            raise RuntimeError(err_msg)
        
        # Write both address 4 and 8 to zero to tell FPGA to attend to the UART states
        self.serial.write(b"W 4 F0000000\n") # Turn vibration off on lasers
        response = self.serial.readline().decode().strip()
        if response != "OK":
            log.error(f"Error: Failed to turn off vibration on lasers: {response}")
            raise RuntimeError(f"Error: Failed to turn off vibration on lasers: {response}")
        self.serial.write(b"W 8 F0000000\n") # Turn off all lasers
        response = self.serial.readline().decode().strip()
        if response != "OK":
            log.error(f"Error: Failed to turn off lasers: {response}")
            raise RuntimeError(f"Error: Failed to turn off lasers: {response}")
        
        ###VIBRATION PERIOD SETTING
        time.sleep(0.1)
        self.serial.write(b"W C AAAAAAAA\n")  #Set PWM period to 16777216 clock cycles (approx 1 second at 16MHz)
        resp = self.serial.readline().decode().strip()
        if resp != "OK":
            log.error(f"Error: Failed to write vibration: {resp}")
            raise RuntimeError(f"Error: Failed to write vibration: {resp}")

        time.sleep(0.1)
        self.serial.write(b"W 10 AAAAAAAA\n")  #Set PWM period to 16777216 clock cycles (approx 1 second at 16MHz)
        resp = self.serial.readline().decode().strip()
        if resp != "OK":
            log.error(f"Error: Failed to write vibration: {resp}")
            raise RuntimeError(f"Error: Failed to write vibration: {resp}")

        time.sleep(0.1)
        self.serial.write(b"W 14 AAAAAAAA\n")  #Set PWM period to 16777216 clock cycles (approx 1 second at 16MHz)
        resp = self.serial.readline().decode().strip()
        if resp != "OK":
            log.error(f"Error: Failed to write vibration: {resp}")
            raise RuntimeError(f"Error: Failed to write vibration: {resp}")

        time.sleep(0.1)
        self.serial.write(b"W 18 AAAAAAAA\n")  #Set PWM period to 16777216 clock cycles (approx 1 second at 16MHz)
        resp = self.serial.readline().decode().strip()
        if resp != "OK":
            log.error(f"Error: Failed to write vibration: {resp}")
            raise RuntimeError(f"Error: Failed to write vibration: {resp}")
        ##END VIBRATION PERIOD SETTING

        #Initialize laser states now that state is known
        log.info("Laser states should now be known.")
        self.laser_states = [0, 0, 0, 0]
        self.vibration_states = [0, 0, 0, 0]
        
    def update_lasers(self, states = None):
        command = b"W 8 "
        if states:
            self.laser_states = states
        command += self.to_hex_command(self.laser_states)
        log.debug(f"Writing laser command: {command}")
        self.serial.write(command + b"\n")
        response = self.serial.readline().decode().strip()
        if response != "OK":
            err_str = f"Error: Failed to write laser command: {command}, response = {response}"
            log.error(err_str)
            raise RuntimeError(err_str)

    def update_vibration(self, states = None):
        command = b"W 4 "
        if states is not None:
            self.vibration_states = states
        command += self.to_hex_command(self.vibration_states)
        log.debug(f"Writing vibration command: {command}")
        self.serial.write(command + b"\n")
        response = self.serial.readline().decode().strip()
        if response != "OK":
            err_str = f"Error: Failed to write vibration command: {command}, response = {response}"
            log.error(err_str)
            raise RuntimeError(err_str)

    def to_hex_command(self, states: list[int]):
        '''Converts the states list to the correct hex command'''
        state_str = '{:x}'.format(int(''.join(map(str, states)), 2)).upper()
        byte_str = bytes("F000000" + state_str, encoding='utf-8')
        return byte_str

class AOPIN(daq.PIN):
    direction = daq.PIN_DIRECTION.OUTPUT
    type = daq.PIN_TYPE.ANALOG
    supported_boards = [EVAL_AD5370,]

    def vout(self, voltage: float):
        '''Sets the analog output voltage of the specified pin
        
            Parameters
            ----------
            pinName: str
                The name of the pin to set to the analog voltage
        '''

        super().vout(voltage)

        # TODO: Finish implementing analog output voltage setting
        self.board: Board
        self.board.serial_vout(self.chnl, voltage)

    def reset(self):
        '''
        Resets the analog output pin to its default state (0V).
        '''
        # log.debug(f"Resetting analog output for pin {self.net_name} to 0V")
        self.vout(0.0)
        
class DIOPIN(daq.PIN):
    direction = daq.PIN_DIRECTION.OUTPUT
    type = daq.PIN_TYPE.DIGITAL
    supported_boards = [VC_709]

    def __init__(self, net_name: str, channel: str):
        super().__init__(net_name, channel)

    def dout(self, value: bool):
        self.board: VC_709
         # TODO: Refactor this for better error checking and more clear ownership of state
        self.board.laser_states[self.chnl] = int(bool(value))
        self.board.update_lasers()
        return

    def reset(self):
        self.dout(False)
        return

def find_boards(repeat = False) -> dict[str, serial.Serial]:
    '''Finds all connected DAQ boards and returns a mapping of their IDs to
        serial connections.

        NOTE: The boards should all be in the handshake state after this 

        Parameters
        ----------
        repeat : bool, optional
            If True, will repeat even `find_boards` was previously called.
            Else will exit if the `_board_dict` is already populated. Default
            is to not repeat.

        Returns
        -------
        dict[str, serial.Serial]
            A dictionary mapping board IDs to their serial connections.
    '''
    if _board_dict and not repeat:
        log.info("DAQ boards already found, skipping search.")
        return _board_dict

    log.info("Searching for connected DAQ boards...")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        try:
            log.debug(f"Trying port {port.device}")
            # Attempt to connect to each port
            ser = serial.Serial(port.device, 115200, timeout=2)
            time.sleep(2) # Give time to reset after opening port

            # Read any initial messages and perform handshake
            while ser.in_waiting:
                log.debug(f"Port {port.device} has {ser.in_waiting} bytes.")
                buffer = ser.read_all() # Clear the buffer
                log.info(f"Pre-handshake buffer contained message: {buffer.decode().strip()}")
                time.sleep(100e-3) # Give time to make sure buffer is clear

            # Step 1: Send handshake request
            ser.write(b"HANDSHAKE\n")
            log.info(f"Sent: 'HANDSHAKE' to {port.device}")

            # Step 2: Read UID from Pico
            uid_response = ser.readline().decode().strip()
            if uid_response.startswith("UID:"):
                board_id = uid_response.split(":")[1]
                log.info(f"Received UID: {board_id}")
            else:
                log.warning(f"Unexpected response during handshake: {uid_response}")
                return None

            if board_id:
                _board_dict[board_id] = ser
            else:
                ser.close() # Close connection if handshake failed
        except serial.SerialException as e:
            if "(5," in str(e.args):
                continue # Ignore input/output errors
            else:
                log.warning(f"Could not connect to {port.device}: {e}")
    return _board_dict

def get_detected_devices():
    """
    Returns a dictionary of all detected DAQ devices.
    """
    find_boards()
    
    if not _board_dict:
        raise Exception("Error: No serial DAQ boards found.")

    device_ids = list(_board_dict.keys())
    for uid in _board_dict:
        print(f"Device ID = {uid}")

def config_detected_devices(boards: list[BoardManager], 
                            verbose: bool = True):
    '''Search the connected DAQ devices and add them to the UL with an assigned
        board number based on their position in the device_ids list.

        Parameters
        ----------
        boards : list[Board]
            A list of Board objects to be configured with the detected devices.
            Each board should have a unique ID that matches the device's serial number.
        verbose : bool, optional
            If True, prints the found devices and their IDs, else they are logged using
            the logger. Default is True.
    '''
    find_boards()
    if not _board_dict:
        log.error("Error: No serial DAQ boards found.")
        raise Exception("Error: No serial DAQ boards found.")
    
    device_ids = [board.board_name for board in boards]

    num_boards = len(boards)
    num_initialized = 0

    if verbose:
        print(f'Found {len(_board_dict)} serial DAQ board(s):')
        for uid in _board_dict:
            print(f"Device ID = {uid}")
            if uid in device_ids:
                num_initialized += 1
    else:
        log.info(f'Found {len(_board_dict)} serial DAQ board(s):')
        for uid in _board_dict:
            log.info(f"Device ID = {uid}")
            if uid in device_ids:
                num_initialized += 1

    for board in boards:
        if not isinstance(board, BoardManager):
            log.error(f"Error: Board of type={type(board)} is not a BoardManager.")
            raise Exception(f"Error: Board of type={type(board)} is not a BoardManager.")

        if board.board_name not in _board_dict:
            log.error(f"Error: Board {board.board_name} not found.")
            raise Exception(f"Error: Board {board.board_name} not found.")

        ser = _board_dict[board.board_name]
        board.serial = ser
        board.board_num = boards.index(board)
        board.initialize_board() # initialize DAC/ADC definitions

    if num_initialized != num_boards:
        log.error(f"Error: {num_boards - num_initialized} boards not initialized. "
                  "Check if the unique IDs exist and are plugged in. "
                  "Use 'ser.get_detected_devices()' to see the available devices.")
        raise Exception(f"Error: {num_boards - num_initialized} boards not initialized. "
                        "Check if the unique IDs exist and are plugged in. "
                        "Use 'ser.get_detected_devices()' to see the available devices.")

# Mapping from unique board IDs to their serial connections
_board_dict: dict[str, serial.Serial] = {}

if __name__ == "__main__":
    dac = EVAL_AD5370("DAC1",   # name
                      0,        # uid
                      17,       # csPin
                      AOPIN("1_1_i", 0),
                      )
    
    # The below dac definitions should trigger an error
    # bad_dac1 = EVAL_AD5370("DAC2",   # name
    #                        0,        # uid
    #                        17,       # csPin # ERROR: SAME CSPIN
    #                        AOPIN("2_2_i", 0))
    # bad_dac2 = EVAL_AD5370("DAC1",   # name # ERROR: DUPLICATE NAME
    #                        0,        # uid
    #                        8,       # csPin # ERROR: SAME CSPIN
    #                        AOPIN("3_3_i", 0))

    pico = BoardManager("PICO-001", # uid
                        dac,
                        # bad_dac1,
                        # bad_dac2,
                        )
    config_detected_devices([pico], verbose=True)
    netlist = daq.Netlist(pico)
    with netlist:
        netlist["1_1_i"].vout(1.5)
        volt = 1.5
        while (True):
            print(f"Sent command for voltage {volt}V (check logs for binary string)")
            volt = input(f"Enter new voltage for pin 1_1_i on {dac.board_name} ('q' to exit): ")
            if volt.lower() in ['q', 'quit', 'exit']:
                print("Exiting...")
                break
            netlist["1_1_i"].vout(float(volt))