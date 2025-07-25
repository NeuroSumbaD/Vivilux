'''Submodule for laser hardware control.
'''

from typing import Any
from time import sleep, time

import numpy as np
from scipy.optimize import curve_fit, fsolve
import nidaqmx
import pyvisa as visa

from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.utils import magnitude
import vivilux.hardware.daq as daq
from vivilux.logger import log

SLEEP = 0.5 # seconds
LONG_SLEEP = 0.5 # seconds

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


        # guarantee control nets are set to 0V
        for net in self.control_nets:
            netlist[net].vout(0.0)

        # Measure offsets for detector readings
        self.initialized_offsets = False
        self.offsets = np.zeros(self.size)
        self.readPhotocurrent()  # Initialize offsets by reading the detectors

        if calibrate:
            self.calibrate_power(calibration_points)
        else:
            self._normalize = lambda x: (x - limits[0]) / (limits[1] - limits[0])
            self._denormalize = lambda x: x * (limits[1] - limits[0]) + limits[0]

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

    def calibrate_power(self, calibration_points: int) -> None:
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
        for voltage in controls:
            self.setControl(np.full(self.size, voltage))
            sleep(10e-3)
            photocurrents.append(self.readPhotocurrent())
            self.setControl(np.zeros(self.size))  # Reset control nets to 0 V
            sleep(100e-3)  # Allow time for the lasers to settle
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

        self._denormalize = lambda x: norm_slopes * x + norm_intercept  # Denormalization function
        self._normalize = lambda x: np.maximum((x - intercepts) / (target_max - intercepts), 0)  # Normalization function

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
        '''
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        # Calculate the denormalized vector and set the power
        self.setControl(self.denormalize(vector))

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
    def __init__(self, port='GPIB0::20::INSTR'):
        #self.wss = visa.SerialInstrument('COM1')
        self.rm = visa.ResourceManager()
        self.main = self.rm.open_resource(port)
        #self.main.baud_rate = 115200
        self.id = self.main.query('*IDN?')
        print(self.id)
        
    def readpow(self,slot=2,channel=1):
        outpt = self.main.query('fetch'+str(slot) +':chan'+str(channel)+':pow?')
        return float(outpt)*1e9 #nW

    def write(self,str):
        self.main.write(str)
        
    def query(self,str):
        result = self.main.query(str)
        return result
    
    def readpowall4(self):
        pow_list = []
        slot_list = [2,4]
        for k in slot_list:
            for kk in range(2):
                pow_list.append(self.readpow(slot=k,channel=kk+1))
        return pow_list
    
    def readpowall2(self):
        pow_list = []
        pow_list.append(self.readpow(slot=4,channel=1))
        pow_list.append(self.readpow(slot=4,channel=2))
        return pow_list
    
    def laserpower(self,inpt):
        self.main.write('sour0:pow '+str(inpt[0])+'uW')
        self.main.write('sour1:pow '+str(inpt[1])+'uW')
        self.main.write('sour3:pow '+str(inpt[2])+'uW')
        self.main.write('sour4:pow '+str(inpt[3])+'uW')
        
    def lasers_on(self,inpt=[1,1,1,1]):
        self.main.write('sour0:pow:stat '+str(inpt[0]))
        self.main.write('sour1:pow:stat '+str(inpt[1]))
        self.main.write('sour3:pow:stat '+str(inpt[2]))
        self.main.write('sour4:pow:stat '+str(inpt[3]))


class InputGenerator:
    def __init__(self, size=4, detectors = [12,8,9,10], limits=[160,350], verbose=False) -> None:
        self.size = size
        self.agilent = Agilent8164()
        self.agilent.lasers_on()
        self.detectors = detectors
        self.limits=limits
        self.maxMagnitude = limits[1]-limits[0]
        self.lowerLimit = limits[0]
        self.chanScalingTable = np.ones((20, size)) #initialize scaling table
        self.calibrated = False
        # self.calibratePower()
        self.readDetectors()
        sleep(SLEEP)
        self.calculateScatter()
        
    def calculateScatter(self):
        print("Calculating the scatter matrix...")
        Y = np.zeros((self.size,self.size))
        Xinv = np.zeros((self.size,self.size))
        for chan in range(self.size):
            oneHot = np.zeros(self.size)
            oneHot[chan] = 1
            print("\tOne-hot: ", oneHot)
            self.agilent.lasers_on(oneHot.astype(int))
            sleep(LONG_SLEEP)
            self.agilent.laserpower(oneHot*350)
            sleep(LONG_SLEEP)
            Y[:,chan] = self.readDetectors()
            Xinv[:,chan] = oneHot/350
            sleep(LONG_SLEEP)
        # print("Detector outputs:\n", Y)
        self.scatter = Y @ Xinv
        # print("Scatter matrix:\n", self.scatter)
        self.invScatter = np.linalg.inv(self.scatter)
        print("Inverse scatter matrix:\n", self.invScatter)
        # maxS = np.max(self.invScatter)
        # minS = np.min(self.invScatter)
        # self.a = 250/(maxS-minS)
        # self.b = 350 - self.a*maxS
        self.a = 350-100
        self.b = 100
        
            

    # def calibratePower(self):
    #     '''Create a table for the scaling factors between power setting and the
    #         true measured values detected on chip.
    #     '''
    #     print("Calibrating power...")
    #     self.powers = np.linspace(0,1,20)

    #     for index, power in enumerate(self.powers):
    #         vector = np.ones(self.size) * power
    #         self.__call__(vector)
    #         detectors = self.readDetectors()
    #         # print(f"\tDetector values: {detectors}")
    #         self.chanScalingTable[index,:] = np.min(detectors)/detectors # scaling factors
    #     print("Scaling table:")
    #     print(self.chanScalingTable)
    #     self.calibrated = True

    # def chanScaling(self, vector):
    #     '''Accounts for differing coupling factors when putting an input into
    #         the MZI mesh. The scaling table must first be generated by running 
    #         `calibratePower()`.
    #     '''
    #     scaleFactors = np.ones(self.size)
    #     for chan in range(self.size):
    #         intendedPower = vector[chan]
    #         scaling = self.chanScalingTable[:, chan]
    #         # Find index for sorted insertion
    #         tableIndex = np.searchsorted(self.powers, intendedPower)
    #         if tableIndex == 0 or tableIndex==len(self.powers):
    #             scaleFactors[chan] = scaling[tableIndex] if tableIndex == 0 else scaling[tableIndex-1]
    #         else:
    #             lowerPower = self.powers[tableIndex-1] 
    #             upperPower = self.powers[tableIndex] 
    #             lowerFactor = scaling[tableIndex-1] 
    #             upperFactor = scaling[tableIndex]
    #             linInterpolation = (intendedPower-lowerPower)/(upperPower-lowerPower)
    #             scaleFactors[chan] = (linInterpolation)*upperFactor + (1-linInterpolation)*lowerFactor
    #     return np.multiply(vector, scaleFactors)

    def scalePower(self, vector):
        '''Takes in a vector with values on the range [0,1] and scales
            according to the max and min of the lasers and the attenuation
            factors between channels.
        '''
        # scaledVector = self.a * (self.invScatter @ vector)
        scaledVector = self.a * vector
        # if self.calibrated:
        #     scaledVector = self.chanScaling(scaledVector)
        return scaledVector + self.b
    
    def readDetectors(self):
        if not hasattr(self, "offset"):
            self.agilent.lasers_on(np.zeros(self.size))
            sleep(LONG_SLEEP)
            with nidaqmx.Task() as task:
                for chan in self.detectors:
                    task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(chan),min_val=-0.0,
                    max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
                data = np.array(task.read(number_of_samples_per_channel=100))
                data = np.mean(data[:,10:],axis=1)
            self.offset = data
            self.agilent.lasers_on(np.ones(self.size))
            sleep(LONG_SLEEP)
        with nidaqmx.Task() as task:
            for chan in self.detectors:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(chan),min_val=-0.0,
                max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            data = np.array(task.read(number_of_samples_per_channel=100))
            data = np.mean(data[:,10:],axis=1)
        return np.maximum((self.offset - data),0)/220*1e3

    def __call__(self, vector, verbose=False, scale=1, **kwds: Any) -> Any:
        '''Sets the lasers powers according to the desired vector
        '''

        vector = self.scalePower(vector)
        vector *= scale
        if verbose: print(f"Inputting power: {vector}")
        self.agilent.laserpower(vector)
        sleep(SLEEP)
        if verbose:
            reading = self.readDetectors()
            print(f"Input detectors: {reading}, normalized: {reading/magnitude(reading)}")


if __name__ == "__main__":
    # Example usage of LaserArray
    from vivilux.hardware.daq import Netlist
    control_nets = ["laser_1", "laser_2"]