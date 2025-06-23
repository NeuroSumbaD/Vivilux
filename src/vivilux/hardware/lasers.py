'''Submodule for laser hardware control.
'''

from typing import Any
from time import sleep

import jax.numpy as jnp
import nidaqmx
import pyvisa as visa

from vivilux.hardware.utils import magnitude
from vivilux.logger import log
import vivilux.hardware.daq as daq

SLEEP = 0.5 # seconds
LONG_SLEEP = 0.5 # seconds

class LaserArray:
    '''Base class for interface with laser arrays for MZI input.
        Assumes that the DAQ voltage output controls the laser power
        (requires voltage controlled current controller circuit).
    '''
    def __init__(self,
                 size: int, # Number of channels in the laser array
                 control_nets: list[str],  # List of control net names to write to
                 detector_nets: list[str],  # List of detector net names to read from
                 limits: tuple[float, float], # (min, max) control signal limits
                 netlist: daq.Netlist,
                 transimpedance: float = 220e3, # transimpedance of the detectors in ohms (default: 220k ohms)
                ):
        if not netlist.in_context:
            log.error("Attempted to initialize LaserArray outside a daq.Netlist context.")
            raise ValueError("LaserArray must be initialized in a netlist context. "
                             "Use `with netlist: \n\t[Network definition and training]` ")

        self.size = size
        self.control_nets = control_nets
        self.detector_nets = detector_nets
        self.limits = limits
        self.netlist = netlist


        self.transimpedance = transimpedance

        # guarantee control nets are set to 0V
        for net in self.control_nets:
            netlist[net].vout(0.0)

        # Measure offsets for detector readings
        self.initialized_offsets = False
        self.offsets = jnp.zeros(self.size)
        self.readPhotocurrent()  # Initialize offsets by reading the detectors

        self.optical_power_scale = None # Scales from power to control signal units (must be calibrated)
        self.calibrate_power_scale()

    def reset(self):
        for net in self.control_nets:
            self.netlist[net].reset()  # Reset control nets to default state (0 V)

    def clip(self, vector: jnp.ndarray) -> jnp.ndarray:
        '''Ensures that the input vector is within the power limits.
        '''
        if not isinstance(vector, jnp.ndarray):
            vector = jnp.array(vector)
        return jnp.clip(vector, self.limits[0], self.limits[1])
    
    def normalize(self, vector: jnp.ndarray) -> jnp.ndarray:
        '''Normalizes the input vector to the range [0, 1].
        '''
        if not isinstance(vector, jnp.ndarray):
            vector = jnp.array(vector)
        return (vector - self.limits[0]) / (self.limits[1] - self.limits[0])
    
    def denormalize(self, vector: jnp.ndarray) -> jnp.ndarray:
        '''Denormalizes the input vector from the range [0, 1] to the control
            unit limits (usually voltage).
        '''
        if not isinstance(vector, jnp.ndarray):
            vector = jnp.array(vector)
        return vector * (self.limits[1] - self.limits[0]) + self.limits[0]
    
    def calibrate_power_scale(self) -> None:
        '''Calibrates the power scale from control signal to optical power.
            This is a placeholder method and should be implemented in subclasses.
        '''
        if self.optical_power_scale is None:
            # Default scale factor, can be overridden in subclasses
            self.optical_power_scale = jnp.zeros(len(self.control_nets), dtype=float)
            for index, net in enumerate(self.control_nets):
                self.netlist[net].vout(self.limits[1]) # set to max power
                # TODO: add sleep if necessary for turn-on time
                # calculate and store a scaling factor 
                self.optical_power_scale = self.optical_power_scale.at[index].set(self.readPhotocurrent()[index] / self.limits[1])
            log.debug(f"Optical power scale set to: {self.optical_power_scale}")

        else:
            log.warning(f"Optical power scale already set to: {self.optical_power_scale}")

    def readPhotocurrent(self) -> jnp.ndarray:
        '''Reads from the detector nets and returns the photocurrent as a numpy array.
        '''
        values = jnp.zeros(self.size)
        for i, net in enumerate(self.detector_nets):
            values = values.at[i].set(self.netlist[net].vin())

        # TODO: refactor to get rid of the if statement (separate offset initialization)
        if not self.initialized_offsets:
            # Initialize offsets if not already done
            self.offsets = values
            self.initialized_offsets = True
            log.debug(f"Initialized offsets: {self.offsets}")

        reading = self.offsets - values  # Subtract offsets to get actual readings
        reading /= self.transimpedance  # Convert to photocurrent (proportional to power)
        # NOTE: most c-band detectors have around 0.9 A/W so photocurrent is
        # pretty close to power in Watts
        return reading
    
    def setControl(self, control_vector: jnp.ndarray) -> None:
        '''Sets the laser powers according to the input vector in terms
            of control signals (usually voltage).
            The input vector should be in the range of self.limits=[min, max]
        '''
        if not isinstance(control_vector, jnp.ndarray):
            control_vector = jnp.array(control_vector)
        # control_vector = control_vector / self.optical_power_scale

        log.debug(f"Setting laser powers to: {control_vector*self.optical_power_scale} (Watts) "
                  f"with control signals: {control_vector} (Volts)")

        for i, net in enumerate(self.control_nets):
            self.netlist[net].vout(float(control_vector[i]))

        # TODO: add sleep if necessary for stable turn-on time

    def setNormalized(self, vector: jnp.ndarray) -> None:
        '''Sets the laser powers from a normalized input vector.
        '''
        if not isinstance(vector, jnp.ndarray):
            vector = jnp.array(vector)
        # Calculate the denormalized vector and set the power
        self.setControl(self.denormalize(vector))

    def display_power_vs_control(self, num_points=100) -> None:
        '''Displays a plot of the power vs control signal for each channel.
            This is useful for visualizing the calibration of the laser array.
        '''
        import matplotlib.pyplot as plt

        control_signals = self.denormalize(jnp.linspace(0, 1, num_points))
        control_signals = jnp.tile(control_signals, (self.size, 1)).T  # Repeat for each channel

        power_readings = jnp.zeros((num_points, self.size))

        for i, control_vector in enumerate(control_signals):

            self.setControl(control_vector)
            power_readings = power_readings.at[i].set(self.readPhotocurrent())



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
        self.chanScalingTable = jnp.ones((20, size)) #initialize scaling table
        self.calibrated = False
        # self.calibratePower()
        self.readDetectors()
        sleep(SLEEP)
        self.calculateScatter()
        
    def calculateScatter(self):
        print("Calculating the scatter matrix...")
        Y = jnp.zeros((self.size,self.size))
        Xinv = jnp.zeros((self.size,self.size))
        for chan in range(self.size):
            oneHot = jnp.zeros(self.size)
            oneHot = oneHot.at[chan].set(1)
            print("\tOne-hot: ", oneHot)
            self.agilent.lasers_on(oneHot.astype(int))
            sleep(LONG_SLEEP)
            self.agilent.laserpower(oneHot*350)
            sleep(LONG_SLEEP)
            Y = Y.at[:,chan].set(self.readDetectors())
            Xinv = Xinv.at[:,chan].set(oneHot/350)
            sleep(LONG_SLEEP)
        # print("Detector outputs:\n", Y)
        self.scatter = Y @ Xinv
        # print("Scatter matrix:\n", self.scatter)
        self.invScatter = jnp.linalg.inv(self.scatter)
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
            self.agilent.lasers_on(jnp.zeros(self.size))
            sleep(LONG_SLEEP)
            with nidaqmx.Task() as task:
                for chan in self.detectors:
                    task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(chan),min_val=-0.0,
                    max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
                data = jnp.array(task.read(number_of_samples_per_channel=100))
                data = jnp.mean(data[:,10:],axis=1)
            self.offset = data
            self.agilent.lasers_on(jnp.ones(self.size))
            sleep(LONG_SLEEP)
        with nidaqmx.Task() as task:
            for chan in self.detectors:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(chan),min_val=-0.0,
                max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            data = jnp.array(task.read(number_of_samples_per_channel=100))
            data = jnp.mean(data[:,10:],axis=1)
        return jnp.maximum((self.offset - data),0)/220*1e3

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