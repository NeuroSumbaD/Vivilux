'''Submodule for laser hardware control.
'''

from time import sleep, time

import numpy as np
from scipy.optimize import curve_fit, fsolve
import pyvisa as visa
import tqdm

from vivilux.hardware.daq import Netlist
from vivilux.hardware.detectors import DetectorArray
import vivilux.hardware.daq as daq
import vivilux.hardware.ser as ser
from vivilux.logger import log

SLEEP = 0.5 # seconds
LONG_SLEEP = 0.5 # seconds

def dBm_to_mW(dBm: float) -> float:
    '''Convert power in dBm to mW.'''
    return 10 ** (np.array(dBm) / 10)

def mW_to_dBm(mW: float) -> float:
    '''Convert power in mW to dBm.'''
    return 10 * np.log10(mW)

class LaserArray:
    '''Base class for interface with laser arrays for MZI input.
        Assumes that the DAQ voltage output controls the laser power
        (requires voltage controlled current controller circuit).
        
        Parameters
        ----------
        size : int
            Number of channels in the laser array.
        control_nets : list[str]
            List of control net names to write to (e.g. "laser_1", "laser_2").
        detectors : DetectorArray
            List of detector net names to read from (e.g. "PD_1_0", "PD_2_0").
        limits : tuple[float, float]
            (min, max) control signal limits (usually voltage) for the laser array.
        netlist : daq.Netlist
            Netlist to use for reading and writing to the control and detector nets.
        transconductance : float
            Transconductance of the laser driver in ohms (default: 480 ohms).
            Lasers are linear in current, so this information helps keep track of all units.
        calibrate : bool
            Whether to run the calibration routine to equalize normalized input ranges.
            If True, the normalized input range will be calibrated based on the
            initial readings such that all lasers give the same power in the
            normalized range.
            If False, the normalized scale will be based on the control limits.
        calibration_points : int
            Number of points to use for calibration (default: 100).
        pause : float
            Pause between control signal changes and read operations (default: 10e-3 seconds).
        duty_cycle : float
            Duty cycle for the calibration routine (default: 0.1).
    '''
    def __init__(self,
                 size: int, # Number of channels in the laser array
                 control_nets: list[str],  # List of control net names to write to
                 detectors: DetectorArray,  # List of detector net names to read from
                 limits: tuple[float, float], # (min, max) control signal limits
                 netlist: daq.Netlist,
                 transconductance: float = 480, # transconductance of the laser driver in ohms (default: 480 ohms)
                 calibrate: bool = True, # whether to run the calibration routine to equalize normalized input ranges
                 calibration_points: int = 100, # number of points to use for calibration
                 pause: float = 10e-3, # pause between control signal changes and read operations
                 duty_cycle: float = 0.1,  # duty cycle for the calibration routine
                ):
        if not netlist.in_context:
            log.error("Attempted to initialize LaserArray outside a daq.Netlist context.")
            raise ValueError("LaserArray must be initialized in a netlist context. "
                             "Use `with netlist: \n\t[Network definition and training]` ")

        self.size = size
        self.control_nets = control_nets
        self.detectors = detectors
        self.limits = limits
        self.netlist = netlist
        self.transconductance = transconductance
        self.pause = pause

        # guarantee control nets are set to 0V
        for net in self.control_nets:
            netlist[net].vout(0.0)

        # Measure offsets for detector readings
        self.initialized_offsets = False
        self.offsets = np.zeros(self.size)
        self.readPhotocurrent()  # Initialize offsets by reading the detectors

        if calibrate:
            self.calibrate_power(calibration_points,
                                 pause,
                                 duty_cycle)
        else:
            def norm(x: np.ndarray) -> np.ndarray:
                return (x - limits[0]) / (limits[1] - limits[0])
            self._normalize = norm
            def denorm(x: np.ndarray) -> np.ndarray:
                return x * (limits[1] - limits[0]) + limits[0]
            self._denormalize = denorm

    def reset(self):
        for net in self.control_nets:
            self.netlist[net].reset()  # Reset control nets to default state (0 V)

    def clip(self, vector: np.ndarray) -> np.ndarray:
        '''Ensures that the input vector is within the power limits.
        '''
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        return np.clip(vector, self.limits[0], self.limits[1])
    
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        '''Normalizes the input vector to the range [0, 1].
        '''
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        # return (vector - self.limits[0]) / (self.limits[1] - self.limits[0])
        return self._normalize(vector)
    
    def denormalize(self, vector: np.ndarray) -> np.ndarray:
        '''Denormalizes the input vector from the range [0, 1] to the control
            unit limits (usually voltage).
        '''
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        # return vector * (self.limits[1] - self.limits[0]) + self.limits[0]
        return self._denormalize(vector)

    def calibrate_power(self,
                        calibration_points: int,
                        pause=10e-3,
                        duty_cycle=0.1) -> None:
        '''Iterates the control nets over the full range in limits and then
            calculates an equalized normal range for all lasers in the array.
            
            Calibration procedure is as follows:
            1. Iterate over the limits of the control nets with `calibration_points`
               number of points and read the photocurrent from the detectors.
            2. Take the data between 10% and 90% of the maximum photocurrent
               for each detector and fit a linear function to the data.
            3. Calculate the exact 90% max from the linear fit, and use the lowest
                maximum as the target maximum for all lasers in the normalized range.
            4. Generate _normalize and _denormalize functions based on the
               slopes from the linear fitting.
        '''
        log.info("Starting calibration of the laser array with nets ")
        start = time()
        # Step 1: iterate over the control limits and read the photocurrent
        photocurrents = []
        controls = np.linspace(self.limits[0], self.limits[1], calibration_points)
        for voltage in tqdm.tqdm(controls,
                                 desc="Calibrating Laser Array",
                                 total=calibration_points,):
            self.setControl(np.full(self.size, voltage))
            sleep(pause)
            photocurrents.append(self.readPhotocurrent())
            self.setControl(np.zeros(self.size))  # Reset control nets to 0 V
            sleep(pause/duty_cycle)  # Allow time for the lasers to settle
        photocurrents = np.array(photocurrents)
        
        # Step 2: Fit a linear function to the data
        slopes = []
        intercepts = []
        for i in range(self.size):
            # Get the data for the i-th detector
            x = controls
            y = photocurrents[:, i]
            
            # isolate 10% to 90% of the maximum range
            max_y = np.max(y)
            mask = np.logical_and(y > (0.1 * max_y), y < (0.9 * max_y))

            # Fit a linear function to the data
            (slope, intercept), pcov = curve_fit(lambda x, m, b: m*x + b,
                                                 x[mask], y[mask],
                                                 )
            slopes.append(slope)
            intercepts.append(intercept)

        # Step 3: Calculate the target maximum for the normalized range
        target_max = 0.9*np.min(np.max(photocurrents, axis=0)) # 90% of the lowest maximum across each channel
        
        # Step 4: Generate the normalization and denormalization functions
        power_vs_input = lambda x: slopes * x + intercepts  # Linear function for power vs control input
        intercept_max = fsolve(lambda x: power_vs_input(x) - target_max, np.zeros(self.size))
        intercept_10 = fsolve(lambda x: power_vs_input(x) - (0.1*target_max), np.zeros(self.size))
        
        norm_slopes = (intercept_max - intercept_10)/(0.9)  # Calculate the slopes for normalization
        norm_intercept = intercept_max - norm_slopes
        
        def norm(x: np.ndarray) -> np.ndarray:
            return np.maximum((x - intercepts) / (target_max - intercepts), 0)  # Normalization function
        def denorm(x: np.ndarray) -> np.ndarray:
            vector = norm_slopes * x + norm_intercept
            vector = np.clip(vector, self.limits[0], intercept_max)  # Ensure within limits
            return vector

        self._denormalize = denorm
        self._normalize = norm

        log.info(f"Calibration complete, took {time() - start:.2f} seconds."
                 f" Normalization slopes: {norm_slopes}")

    def readPhotocurrent(self) -> np.ndarray:
        '''Reads from the detector nets and returns the photocurrent as a numpy array.
        '''
        return self.detectors.read()  # Use the DetectorArray's read method to get photocurrent
    
    def setControl(self, control_vector: np.ndarray) -> None:
        '''Sets the laser powers according to the input vector in terms
            of control signals (usually voltage).
            The input vector should be in the range of self.limits=[min, max]
        '''
        if not isinstance(control_vector, np.ndarray):
            control_vector = np.array(control_vector)

        log.info(f"Setting laser control nets to: {control_vector} (Volts)")

        for i, net in enumerate(self.control_nets):
            if control_vector[i] >= self.limits[0] and control_vector[i] <= self.limits[1]:
                self.netlist[net].vout(control_vector[i])
            else:
                log.error(f"Control signal {control_vector[i]} for net {net} is out of bounds ")
                raise ValueError(f"Control signal {control_vector[i]} for net {net} is out of bounds ")

        # TODO: add sleep if necessary for stable turn-on time

    def setNormalized(self, vector: np.ndarray) -> None:
        '''Sets the laser powers from a normalized input vector.
        
            NOTE: This method is used after the calibration routine
            and uses a settling delay to allow the lasers to stabilize.
        '''
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        # Calculate the denormalized vector and set the power
        self.setControl(self.denormalize(vector))
        sleep(self.pause)  # Allow time for the lasers to settle

    def display_power_vs_control(self, num_points=100) -> None:
        '''Displays a plot of the power vs control signal for each channel.
            This is useful for visualizing the calibration of the laser array.
        '''
        import matplotlib.pyplot as plt

        control_signals = self.denormalize(np.linspace(0, 1, num_points))
        control_signals = np.tile(control_signals, (self.size, 1)).T  # Repeat for each channel

        power_readings = np.zeros((num_points, self.size))

        for i, control_vector in enumerate(control_signals):

            self.setControl(control_vector)
            power_readings[i] = self.readPhotocurrent()



class Agilent8164():
    '''Outdated class for controlling the Agilent 8164 laser array.
        This class uses the PyVISA library to communicate with the laser
        controller over GPIB or serial connection.

        TODO: Update to be compliant with the new LaserArray interface.
    '''
    def __init__(self,
                 port='GPIB0::20::INSTR',
                 channels=[1, 2, 3, 4]
                 ):
        #self.wss = visa.SerialInstrument('COM1')
        self.rm = visa.ResourceManager()
        log.info(f"Found visa resources: {self.rm.list_resources()}")
        self.main: visa.Resource = self.rm.open_resource(port)
        #self.main.baud_rate = 115200
        self.id: str = self.main.query('*IDN?')
        log.info(f"Agilent8164 ID: {self.id.strip()}")
        self.channels = channels
        
    def _to_bool(self, resp: str) -> bool:
        s = resp.strip().upper()
        return s.startswith('1') or s.startswith('ON') or s.startswith('TRUE')
        
    def readpow(self,slot=2,channel=1):
        outpt = self.main.query('fetch'+str(slot) +':chan'+str(channel)+':pow?')
        return float(outpt)*1e9 #nW
    
    def isOn(self, channel: int) -> bool:
        '''Checks if the laser on the specified channel is turned on.
        '''
        response = self.query(f"sour{channel}:pow:stat?")
        # log.debug(f"Response while checking channel {channel} (pow:stat?)): {response.strip()}")
        response = self._to_bool(response)
        response2 = self.query(f"outp{channel}:stat?")
        # log.debug(f"Response while checking channel {channel} (outp:stat?): {response2.strip()}")
        response2 = self._to_bool(response2)
        response = response or response2
        return response

    def allOn(self) -> bool:
        '''Checks if all lasers are turned on.
        '''
        return all(self.isOn(chan) for chan in self.channels)

    def write(self,str):
        self.main.write(str)
        
    def query(self,str):
        result = self.main.query(str)
        return result

    def laserpower(self, mW: list[float]):
        '''Sets the power of each laser in mW
        '''
        for i, chan in enumerate(self.channels):
            # log.debug(f"Writing the following command to channel {chan}: sour{chan}:pow {mW[i]}mW")
            self.main.write(f'sour{chan}:pow {mW[i]}mW')

    def lasers_on(self, value: list[bool]):
        for i, chan in enumerate(self.channels):
            self.main.write(f'sour{chan}:pow:stat {value[i]}')
            
    # def __del__(self):
    #     '''Destructor to turn off the lasers when the object is deleted.
    #     '''
    #     self.lasers_on([0]*len(self.channels))  # Turn off all lasers
    #     log.info("Agilent laser off signal sent.")
    #     sleep(LONG_SLEEP)  # Allow time for the lasers to turn off
        

class AgilentLaserArray(LaserArray):
    def __init__(self,
                 size: int,
                 detectors: DetectorArray, #[12,8,9,10],
                 netlist: Netlist,
                 upperLimits: np.ndarray = dBm_to_mW(np.array([-5, -3, -5, -9])),
                 lowerLimits: np.ndarray = dBm_to_mW(np.array([-10, -10, -10, -10])),
                 port: str ='GPIB0::20::INSTR',
                 channels: list[int] = [1, 2, 4, 3], # Agilent channels to use (order preserved calls to turn on/off)
                 pause = 100e-3, # Delay before reading the detectors
                 wait = 5, # Wait time after turning on the lasers
                 max_retries = 5, # Number of retries for turning on lasers
                 calibrate = False, # Whether to do normalization calibration
                 ) -> None:
        self.size = size
        self.agilent = Agilent8164(port, channels=channels)
        self.detectors = detectors
        self.channels = channels
        
        self.lowerLimits = lowerLimits
        self.upperLimits = upperLimits
        self.maxMagnitude = upperLimits - lowerLimits
        
        self.powers = lowerLimits
        self.pause = pause
        self.wait = wait
        self.max_retries = max_retries
        
        self.netlist = netlist
        def lasers_off():
            self.setMinimum()
            self.agilent.lasers_on([0]*len(self.channels))
        self.netlist.exit_queue.append(lasers_off)

        # Set the lasers to their minimum power and turn them on
        self.setControl(self.powers)
        self.lasers_on()  # Turn on all lasers
        
        def fromNormal(vector: np.ndarray) -> np.ndarray:
            '''Converts a normalized vector to mW using the upper limits.
            '''
            return self.lowerLimits + vector * self.maxMagnitude
        
        if calibrate:
            log.warning("Calibration mode enabled, this reduces the maximum "
                        "power of each channel to the lowest channel maximum")
            mw = fromNormal(np.ones(self.size))  # Set to maximum power
            
            self.setControl(mw)  # Set the lasers to maximum power
            max_readings = self.detectors.read()
            
            self.setControl(self.lowerLimits)
            min_readings = self.detectors.read()
            
            readings = max_readings - min_readings  # Calculate the readings range
            
            scale_factors = np.min(readings) / readings
            
            def scaledNormal(vector: np.ndarray) -> np.ndarray:
                '''Scales the normalized vector to the maximum power.
                '''
                return self.lowerLimits + vector * self.maxMagnitude * scale_factors
            
            self._fromNormal = scaledNormal
        else:
            self._fromNormal = fromNormal
    
    def readPhotocurrent(self):
        '''Reads the photocurrent from the DetectorArray.
        
            NOTE: For the Agilent lasers, turning off the laser completely
            is slow, so it is better to use the minimum power as the off state
            for all the lasers and create a new offset during each reading. It
            is not possible to calibrate this offset at the beginning of a run
            because the calibration will change the total power reaching each
            detector at their minimum power setting.
        '''
        return self.detectors.read()

    def setControl(self, control_vector: np.ndarray):
        '''Sets the laser powers according to the input vector in terms
            of control signals (usually voltage).
            The input vector should be in the range of self.limits=[min, max]
        '''
        if not isinstance(control_vector, np.ndarray):
            control_vector = np.array(control_vector)

        log.info(f"Setting laser control nets to: {control_vector} (mW)")

        self.powers = control_vector
        self.agilent.laserpower(control_vector)

    def setLast(self):
        '''Sets the laser powers to their previous power setting.
        '''
        self.agilent.laserpower(self.powers)
        sleep(self.pause)  # Allow time for the lasers to settle

    def setMinimum(self):
        '''Sets the laser powers to their minimum values.
        '''
        self.agilent.laserpower(self.lowerLimits)
        sleep(self.pause)  # Allow time for the lasers to settle

    def setNormalized(self, vector: np.ndarray):
        '''Sets the laser powers using normalized values.
        
            NOTE: The lasers do not have similar power levels, so for now each
            one is scaled according to its own maximum power level. This is only
            sufficient when using the lasers for one-hot patterns, and should be
            improved for more complex analog patterns.

            TODO: Create a calibration procedure to equalize the maximum normalized
            power levels for each of the lasers.
        '''
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        if vector.max() > 1 or vector.min() < 0:
            log.warning(f"Attempted to input an un-normalized vector: {vector}")
        vector = np.clip(vector, 0, 1)
        
        mW  = self._fromNormal(vector)
        self.setControl(mW)
        sleep(self.pause)  # Allow time for the lasers to settle
        
    def lasers_on(self,):
        '''Checks if the lasers are turned off and turns them on if necessary.
            Uses recursion to try again, and will error out after max_retries.
        '''
        channel_states = [False] * self.size
        for i in range(self.max_retries):
            for index, channel in enumerate(self.channels):
                if not self.agilent.isOn(channel):
                    channel_states[index] = False
                    if i == 0:
                        self.agilent.lasers_on([1]*self.size)  # Turn on all lasers
                        log.info(f"One or more lasers were not turned on (checked channel: {self.channels})"
                             f" waiting {self.wait} seconds and trying again...")
                        sleep(self.wait)
                        continue
                    if i == self.max_retries - 1:
                        log.error(f"Laser on channel {channel} did not turn on correctly after {self.max_retries} attempts.")
                        log.info(f"Laser powers are set to {self.powers} mW (lowering the power may improve stability)")
                        raise RuntimeError(f"Laser on channel {channel} did not turn on correctly after {self.max_retries} attempts.")
                    else:
                        self.agilent.lasers_on([1]*self.size)
                        log.warning(f"Laser on channel {channel} did not turn on correctly."
                                    f" Retrying {i+1}/{self.max_retries} and waiting {self.wait} seconds...")
                        sleep(self.wait)
                        continue
                else:
                    channel_states[index] = True
            if all(channel_states):
                break

class AgilentDetectorArray(DetectorArray):
    '''A subclass of DetectorArray for working with the Agilent lasers, since they
        cannot be turned off completely without a long delay, each measurement
        requires a new offset to subtract the minimum power level.
    '''
    def __init__(self,
                 detectors: DetectorArray,
                 lasers: AgilentLaserArray,
                 ):
        if not isinstance(detectors, DetectorArray):
            raise TypeError("detectors must be an instance of DetectorArray")
        self.detectorArray = detectors
        self.size = detectors.size
        
        if not isinstance(lasers, AgilentLaserArray):
            raise TypeError("lasers must be an instance of AgilentLaserArray")
        self.lasers = lasers

    def read(self):
        '''In this subclass, the reading needs to substract the minimum power
            from the current reading.
        '''
        self.lasers.lasers_on()

        on_reading = self.detectorArray.read()  # Call the base class read method
        log.info(f"Detector reading with lasers on (mA): {on_reading*1e6}")
        
        # turn the lasers to their minimum power
        self.lasers.setMinimum()
        
        off_reading = self.detectorArray.read()  # Read the detectors with lasers off
        log.info(f"Detector reading with lasers off (mA): {off_reading*1e6}")

        self.lasers.setLast()

        return on_reading - off_reading
    
    def __len__(self):
        return self.detectorArray.size
    
class SFPLaserArray(LaserArray):
    '''A subclass of LaserArray for controlling SFP digital lasers.
    '''
    def __init__(self,
                 size: int,
                 control_nets: list[str],
                 detectors: DetectorArray,
                 netlist: daq.Netlist,
                 board: ser.VC_709,
                 use_vibrations: bool = True, # Whether to use vibrations for laser "on" state
                 pause: float = 100e-6, # pause between control signal changes and read operations
                ):
        if not netlist.in_context:
            log.error("Attempted to initialize LaserArray outside a daq.Netlist context.")
            raise ValueError("LaserArray must be initialized in a netlist context. "
                             "Use `with netlist: \n\t[Network definition and training]` ")

        self.size = size
        self.control_nets = control_nets
        self.detectors = detectors
        self.netlist = netlist
        self.board = board
        
        self.use_vibrations = use_vibrations
        self.pause = pause
        
        self.board.update_vibration([0]*self.size)  # Set all vibrations to off
        if self.use_vibrations:
            log.info("Using vibrations to control SFP laser array.")
            self.board.update_lasers([1]*self.size)  # Ensure all lasers are on initially
            print("Waiting 5 seconds for SFP lasers to turn on...")
            sleep(5)
        else:
            log.info("Using direct laser control for SFP laser array.")
            self.board.update_lasers([0]*self.size)  # Ensure all lasers are off initially
        
        
    def setNormalized(self, vector):
        if any([v not in [0, 1] for v in vector]):
            err_str = f"ERROR: SFP Lasers do not support non-binary values: {vector}."
            log.error(err_str)
            raise ValueError(err_str)
        vector = [int(np.round(v)) for v in vector]
        if self.use_vibrations:
            self.board.update_vibration(vector)
        else:
            self.board.update_lasers(vector)
        sleep(self.pause)  # Allow time for the lasers to settle
            
class SFPDetectorArray(DetectorArray):
    '''A subclass of DetectorArray for working with the Agilent lasers, since they
        cannot be turned off completely without a long delay, each measurement
        requires a new offset to subtract the minimum power level.
    '''
    def __init__(self,
                 detectors: DetectorArray,
                 lasers: SFPLaserArray,
                 ):
        if not isinstance(detectors, DetectorArray):
            raise TypeError("detectors must be an instance of DetectorArray")
        self.detectorArray = detectors
        self.size = detectors.size

        if not isinstance(lasers, SFPLaserArray):
            raise TypeError("lasers must be an instance of SFPLaserArray")
        self.lasers = lasers
        
        self.transimpedance = detectors.transimpedance

    def read(self):
        '''In this subclass, the reading needs to substract the minimum power
            from the current reading.
        '''
        on_reading = self.detectorArray.read()  # Call the base class read method
        log.info(f"Detector reading with lasers on (mA): {on_reading*1e6}")

        # if using vibrations, turn the lasers to their minimum power
        if self.lasers.use_vibrations:
            laser_states = self.lasers.board.vibration_states.copy()
        else:
            return on_reading
        
        self.lasers.setNormalized([0]*self.size)  # Turn off all lasers
        off_reading = self.detectorArray.read()  # Read the detectors with lasers off
        log.info(f"Detector reading with lasers off (mA): {off_reading*1e6}")

        self.lasers.setNormalized(laser_states)  # Restore previous laser states

        return on_reading - off_reading
    
    def read_raw(self):
        on_reading, on_dev = self.detectorArray.read_raw()  # Call the base class read method
        log.info(f"Detector reading with lasers on (mA): {on_reading*1e6}")
        
        # turn the lasers to their minimum power
        if self.lasers.use_vibrations:
            laser_states = self.lasers.board.vibration_states.copy()
        else:
            return on_reading, on_dev
        
        self.lasers.setNormalized([0]*self.size)  # Turn off all lasers
        off_reading, off_dev = self.detectorArray.read_raw()  # Read the detectors with lasers off
        log.info(f"Detector reading with lasers off (mA): {off_reading*1e6}")

        self.lasers.setNormalized(laser_states)  # Restore previous laser states

        return on_reading - off_reading, np.max([on_dev, off_dev], axis=0)
        

if __name__ == "__main__":
    # Example usage of LaserArray
    from vivilux.hardware.daq import Netlist
    control_nets = ["laser_1", "laser_2"]