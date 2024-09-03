# Documentation

## Net

A `Net` is a collection of neural network `Layers` and configuration parameters which define in what sequence various inputs are applied to layers within the network.

## Layer

A `Layer` is an abstraction representing a set of neurons. Layer objects are responsible for applying dynamic nonlinearities within the neural network. A dynamic system is formed based on the processes within 

### NeuronDevice

A `NeuronDevice` is a subclass of `Layer` which implements additional features for defining device architecture and performing energy accounting. Also allows for translation from normalized units used by Leabra to physically meaningful units.

*Note*: This subclass has not yet been implemented

### Processes

Each layer transforms its inputs based on a number of processes. Processes are organized into two temporal domains, `Cyclic` and `Phasic` which correspond to once each time step or once each phase.

*Note*: Minimal documentation is provided in the README.md though some details in implementation may change.

## Path

A `Path` is an abstraction representing a set of synaptic connections between two layers and parameters which define how the synaptic variables are updated during training.

#### Methods:
 - `__init__(sendLayer, recvLayer, **kwargs)`:  initializes the Path object
   - Args:
     - `sendlayer`: pre-synaptic layer.
     - `recvLayer`: post-synaptic layer.
     - `kwargs`:  dict to hold additional parmeters including the traditional leabra parameters for initializing and updating synaptic connections.
 - `Update()`:  Uses the XCAL process to calculate changes to the synaptic matrix in the form of a delta vector in the same shape as the path. Calls an `ApplyUpdate(delta)` that should be overwritten in all subclasses to calculate updates from parameters.
 - `Apply()`:  Calculates and returns synaptic currents for the next layer using data from the previous layer as input.

 ### SynapticDevice(Path)

 A `SynapticDevice` is a subclass of `Path` which adds functionality for energy accounting that integrates the power consumption over the course of simulation. The energy is reported as total energy, static energy (meaning the energy of maintaining the weights), and training energy (energy consumed while updating parameters).

 #### Methods
 - `__init__(sendLayer, recvLayer, DataSignalType=Data, **kwargs)`:  initializes the Path object
   - Args:
     - `sendlayer`: pre-synaptic layer.
     - `recvLayer`: post-synaptic layer.
     - `inDataType`: Specifies the type of data signal that can be received by the synaptic device.
     - `outDataType`: Specifies the type of data signal that is output by the synaptic device.
     - `kwargs`:  dict to hold parmeters for the parent `Path` constructor, but also to hold device parameters for initializing the structure of the current `SynapticDevice`
 - `CalcUpdate(delta: np.ndarray)`:  Calculates the necessary changes in device parameters to implement the change from the delta vector. Should also call necessary functions for calculating the energetic cost of training.
 - `ApplyTo(data: Data)`:  Calculates and returns the synaptic currents generated from applying the matrix to the input. Should also call any functions necessary for calculating hysteresis of memristive effects of underlying parameters.

 #### Subclasses
  - `MZImesh`:  Meshes of photonic interferometers
    - `ClementsMZI`
    - `DiagMZI`
    - `SVDMZI`
  - `Crossbar`:  Simple crossbar arrays
    - `PCMcrossbar`
    - `ReRam`
    - `Ionic`

 ## Signal

 A `Signal` is an abstraction for the data that moves through the neural network and includes features for type checking and conversion between normalized and physically meaningful units. Divided into `Data` signals that communicate the activity of the neural network between layers and synaptic pathways, and `Control` signals which are used to configure parameters of the devices in the network.

 *

 #### Methods
 - `__init__(values=None, shape=None, limits=(0,np.inf), scale=None, dimNames=None, bitPrecision=None, noisefunc = lambda x:x, **kwargs)`
    - Args:
      - `values`: (optional) set of initial values for the signal.
      - `shape`: (optional) shape to initialize signal (required if values are not given).
      - `limits`: (optional) limits to describe the maximum and minimium valeus that the signal can take.
      - `scale`: (optional) scale factor to be used converting between normalized and phyiscally meaningful values. Scale must be provided if limits do not give a finite range.
      - `dimNames`: (optional) metadata to describe what each dimension in the shape represents. e.g. a photonic signal may be composed of multiple waveguides, wavelengths, modes, etc.
      - `bitPrecision`: by default bitPrecision is unrestricted (i.e. infinite), but can be set to discretize the values that can be taken within the limited range of the signal. Cannot be set if finite limits are not provided.
      - `noisefunc`: (optional) function for generating noise. Gets applied on top of the signal each time the `Get()` method is called. Subclasses and calling classes may provide this function.
      - `**kwargs`: catchall for additional parameters that subclasses might use that are ignored in the base class.
 - `Set(data: np.ndarray, record=False)`: updates all representations of the Signal using the physically meaningful representation. Optionally adds the signal to the `records` attribute which tracks the history of signals for calculating various metrics at the end of the simulation.
 - `SetNorm(data: np.ndarray, record=False)`: updates all representations of the Signal using a normalized representation. Optionally adds the signal to the `records` attribute which tracks the history of signals for calculating various metrics at the end of the simulation.
 - `Get()`: returns the physically meaningful representation of the Signal.
 - `GetNorm()`: returns the normalized representation of the Signal.

 #### Attributes
  - `records = []`: a list of previous values the signal has taken, useful for calculating 

 #### Subclasses
  - `Control`: Signals meant for the control plane of the neural network. Used to set tunable device parameters or maintain state.
  - `Data`: Signals meant to represent neural activity in the network and yield interpretable results.
  - `Voltage[Data|Control]`: includes both Data and Control signal subclasses for type-checking functionality
  - `Current[Data|Control]`: includes both Data and Control signal subclasses for type-checking functionality
  - `Photonic[Data|Control]`: primarily represented as complex E-field coefficients for each wavelength with convenience functions for calculating power and phase. Includes both Data and Control signal subclasses for type-checking functionality
 
 ## ParamSet
 
 A `ParamSet` is an abstraction for calculating transfer matrices of devices. Used to provide type-checking features.

#### Methods
 - `__init__(values=None, shape=None, dimNames=None, generator=None, **kwargs)`
    - Args:
      - `values`: optional values to initialize the set of parameters.
      - `shape`: if values are not provided, the shape and generator should be set to initialize a vector of the appropriate shape.
      - `generator`: if values are not provided, the generator function is called which should take in a tuple representing the shape, and 
      - `dimNames`: metadata to describe what each dimension in the shape represents. e.g. a photonic signal may be composed of multiple waveguides, wavelengths, modes, etc.
      - `**kwargs`: catchall for additional parameters that subclasses might use that are ignored in the base class.
 - `Set(data: np.ndarray)`:
 - `Get()`: returns the parameters. Other get functions may return the same parameters in a different representation (e.g. complex phase representation vs radians).


 #### Attributes
  - `records = []`: a list of previous values the parameter has taken, useful for calculating 

 #### Subclasses
  - `PhaseShift`: phase shift that can be represented either as a complex coefficient of a signal or a real number format specifying the phase shift in radians.
  - `Gain`: signal gain represented as a normalized values (e.g. 1.2 for 20% gain). Also allows for conversion to dB.
  - `Attenuation`: signal attenuation represented as a normalized values (e.g. 0.8 for 20% attenuation). Also allows for conversion to dB. for resistance, optical absorber, etc.
  - `Const`: subclasses that should throw an error if modification is attempted, meant to represent random fabrication tolerances that modify some parameter just once and then should not.
    - `CouplingCoeff`: coupling factor between two ports represented as normalized values and with a second dimension to specify the compliment. The shape is stored as (2, numCouplers). e.g. two couplers could be represented as [[0.2,0.4],[0.8,0.6]].
  

 ## MaterialSet

 A `MaterialSet` is an abstraction for parameters that are calculated from equations describing physical interactions. Each `MaterialSet` should be associated with one or more `ParamSet` that describes the parameters that should be returned. The `MaterialSet` should also provide functions for calculating the power consumption required to maintain the state and update the parameters.

 #### Methods
 - `__init__(shape=None, dimNames=None, **kwargs)`
    - Args:
      - `shape`: the shape of underlying `ParamSet` that is returned by the `MaterialSet`.
      - `dimNames` metadata for the name of r
      - `**kwargs`: catchall for additional parameters that subclasses might use that are ignored in the base class.
 - `GetParams()`: returns the `ParamSet` described by the materials.
 - `CalcEnergy(deltaTime: float)`: returns a report of the energy consumption over the simulation using the data stored in the `records` attribute. Returns TotalEnergy, StaticEnergy (consumed by maintaining the state during inference), and TrainingEnergy (consumed during the updating of parameters).
 - `ApplyControl(signal: Control)`: Updates the underlying `ParamSet` according to any changes in the control signal. This function is primarily used during training.
 - `ApplyData(signal: Data)`: Updates the underlying `ParamSet` according to any changes caused by data passing through the material. This is primarily used for simulating the effects of hysteresis.

 #### Attributes
  - `records = {}`: a dictionary of lists recording various states of important variables used over the course of the simulation.
  - `ParamType`: Type of parameter the `MaterialSet` should return, used when type-checking connections between components.

 #### Subclasses
  - `PhaseShifter`: various types of optical phase shifters
    - `ThermalPS`
    - `MOSCAP`
    - `PCM`
  - `Resistive`: elements with tunable resistance controlled by a gate (like a transistor). Also includes attenuative elements like those used in optical crossbars.
  - `Memristor`: (not yet implemented) resistive elements with hysteresis in the data pathway
  