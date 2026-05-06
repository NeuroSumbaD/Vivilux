'''Defines a Net class corresponding to a group of layers interconnected with
    various meshes.
'''

# type checking
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .layers import Layer
    from .meshes import Mesh

import numpy as np
import json

# import defaults
from .meshes import Mesh
from .metrics import RMSE
from .visualize import Monitor

###<------ DEFAULT CONFIGURATIONS ------>###

layerConfig_std = {
    # "DELTA_Vm" : 0.1/2.81,
    "VmInit": 0.4,
    "hasInhib" : True,
    "Gbar": { # Max conductances for each effective channel
        "E": 1.0, # Excitatory
        "L": 0.1, # Leak
        "I": 1.0, # Inhibitory
        "K": 1.0, # Frequency adaptation potassium channel
    },
    "Erev": { # Reversal potential for each effective channel
        "E": 1.0, # Excitatory
        "L": 0.3, # Leak
        "I": 0.25, # Inhibitory
        "K": 0.25, # Frequency adaptation potassium channel
    },
    "DtParams": {
        "Integ" : 1, # overall rate constant for numerical integration, for all equations at the unit level -- all time constants are specified in millisecond units, with one cycle = 1 msec -- if you instead want to make one cycle = 2 msec, you can do this globally by setting this integ value to 2 (etc).  However, stability issues will likely arise if you go too high.  For improved numerical stability, you may even need to reduce this value to 0.5 or possibly even lower (typically however this is not necessary).  MUST also coordinate this with network.time_inc variable to ensure that global network.time reflects simulated time accurately
        "VmTau" : 3.3, # membrane potential and rate-code activation time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) -- reflects the capacitance of the neuron in principle -- biological default for AdEx spiking model C = 281 pF = 2.81 normalized -- for rate-code activation, this also determines how fast to integrate computed activation values over time
        "GTau" : 1.4, # time constant for integrating synaptic conductances, in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) -- this is important for damping oscillations -- generally reflects time constants associated with synaptic channels which are not modeled in the most abstract rate code models (set to 1 for detailed spiking models with more realistic synaptic currents) -- larger values (e.g., 3) can be important for models with higher conductances that otherwise might be more prone to oscillation.
        "AvgTau" : 200, # for integrating activation average (ActAvg), time constant in trials (roughly, how long it takes for value to change significantly) -- used mostly for visualization and tracking *hog* units
        
    },
    "ActAvg": {
        "Init": 0.15,
        "Fixed": False,
        "SSTau": 2,
        "STau": 2,
        "MTau": 10,
        "Tau": 10,
        "AvgL_Init": 0.4,
        "Gain": 2.5,
        "Min": 0.2,
        "LrnM": 0.1,
        "ModMin": 0.01,
        "LrnMax": 0.5,
        "LrnMin": 0.0001,
        "UseFirst": True,
        # "ActPAvg_Init": 0.15,
        "ActPAvg_Tau": 100,
        "ActPAvg_Adjust": 1,
    },
    "OptThreshParams":{
        "Send": 0.1,
        "Delta": 0.005,
    },
    "XCALParams": {
        "DRev": 0.1,
        "DThr": 0.0001,
        "hasNorm": True,
        "Norm_LrComp": 0.15,
        "normMin": 0.001,
        "DecayTau": 1000,
        "hasMomentum": True, #TODO allow this to be set by layer or mesh config
        "MTau": 10, #TODO allow this to be set by layer or mesh config
        "Momentum_LrComp": 0.1, #TODO allow this to be set by layer or mesh config
        "LrnThr": 0.01,
        "Lrate": 0.04,
    },
    "defMonitor": Monitor,
    "FFFBparams": {
        "Gi": 1.8, # [1.5-2.3 typical, can go lower or higher as needed] overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly
        "FF": 1, # overall inhibitory contribution from feedforward inhibition -- multiplies average netinput (i.e., synaptic drive into layer) -- this anticipates upcoming changes in excitation, but if set too high, it can make activity slow to emerge -- see also ff0 for a zero-point for this value
        "FB": 1, # overall inhibitory contribution from feedback inhibition -- multiplies average activation -- this reacts to layer activation levels and works more like a thermostat (turning up when the 'heat' in the layer is too high)
        "FBTau": 1.4, # time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) for integrating feedback inhibitory values -- prevents oscillations that otherwise occur -- the fast default of 1.4 should be used for most cases but sometimes a slower value (3 or higher) can be more robust, especially when inhibition is strong or inputs are more rapidly changing
        "MaxVsAvg": 0, # what proportion of the maximum vs. average netinput to use in the feedforward inhibition computation -- 0 = all average, 1 = all max, and values in between = proportional mix between average and max (ff_netin = avg + ff_max_vs_avg * (max - avg)) -- including more max can be beneficial especially in situations where the average can vary significantly but the activity should not -- max is more robust in many situations but less flexible and sensitive to the overall distribution -- max is better for cases more closely approximating single or strictly fixed winner-take-all behavior -- 0.5 is a good compromise in many cases and generally requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0
        "FF0": 0.1, # feedforward zero point for average netinput -- below this level, no FF inhibition is computed based on avg netinput, and this value is subtraced from the ff inhib contribution above this value -- the 0.1 default should be good for most cases (and helps FF_FB produce k-winner-take-all dynamics), but if average netinputs are lower than typical, you may need to lower it
    },
}

phaseConfig_std = {
    "minus": {
        "numTimeSteps": 75,
        "isOutput": True,
        "isLearn": False,
        "clampLayers": {"input": 0,
                    },
    },
    "plus": {
        "numTimeSteps": 25,
        "isOutput": False,
        "isLearn": True,
        "clampLayers": {"input": 0,
                    "target": -1,
                    },
    },
    "_plus": {
        "numTimeSteps": 25,
        "isOutput": False,
        "isLearn": False,
        "clampLayers": {"input": 0,
                    "target": -1,
                    },
    },
}

runConfig_std = {
    "DELTA_TIME": 0.001,
    "metrics": {
        "RMSE": RMSE,
    },
    "outputLayers": {
        "target": -1,
    },
    "Learn": ["minus", "plus"],
    "Infer": ["minus", "_plus"], # note: the "_" prefix indicates that this pseudo-plus phase is only for inference, and learning is disabled
    "End": {
        "threshold": 0,
        "isLower": True,
        "numEpochs": 5,
    }
}


class Net(ABC):
    '''Abstract base class for neural networks in this repo. Defines the basic
        structure and methods for a network, but does not implement and specific
        architecture or learning rule.
    '''
    @abstractmethod
    def Learn(self, input: np.ndarray,
              target: np.ndarray,
              numEpochs: int,
              **kwargs):
        raise NotImplementedError("Learn method not implemented for base Net class")

    @abstractmethod
    def Evaluate(self, input: np.ndarray, target: np.ndarray, **kwargs):
        raise NotImplementedError("Evaluate method not implemented for base Net class")
    
    @abstractmethod
    def Infer(self, input: np.ndarray, **kwargs):
        raise NotImplementedError("Infer method not implemented for base Net class")


###<------ NET CLASSES ------>###
class LeabraNet(Net):
    '''Instantiation of mechanisms from the Leabra framework with a more clear
        delineation of matrix-vector operations implemented as linear algebraic
        operations rather than individual lists of connections. This takes
        advantage of SIMD parallelism and also makes it easier to implement
        physically realized implementations with abstracted synaptic matrices
        and layers of neurons.

        TODO: Implement definition of multiple input or target layers to allow
        for multi-modal learning. This was initially confirable in the runConfig,
        but is being removed for a more clear state-machine flow.
    '''
    count = 0
    def __init__(self,
                 name = None,
                 monitoring = False,
                 runConfig = runConfig_std,
                 phaseConfig = phaseConfig_std,
                 layerConfig = layerConfig_std,
                 dtype = np.float64,
                 **kwargs):
        '''Instanstiates an ordered list of layers that will be
            applied sequentially during inference.
        '''

        self.runConfig = runConfig # Simulation-level configurations
        self.phaseConfig = phaseConfig # Dictionary of phase definitions
        self.layerConfig = layerConfig # Stereotyped layer definition
        self.monitoring = monitoring
        self.dtype = dtype

        # For time keeping
        self.DELTA_TIME = runConfig["DELTA_TIME"]
        self.time = 0

        self.layers: list[Layer] = [] # list of layer objects
        self.layerDict: dict[str, Layer] = {} # dict of named layers

        self.results = {metric: [] for metric in self.runConfig["metrics"]}
        self.outputs = {key: [] for key in self.runConfig["outputLayers"]}

        self.name =  f"NET_{Net.count}" if name is None else name
        self.epochIndex = 0
        LeabraNet.count += 1

        # For early stopping learning process
        self.lrnThresh = 0

    def PreallocateResultDict(self):
        '''Pre-allocate a dict to store the results
        '''
                
    def AddLayer(self, layer: Layer, layerConfig: dict = None):
        # index = len(self.layers)
        # size = len(layer)

        if layer.dtype != self.dtype:
            layer.SetDtype(self.dtype)

        self.layers.append(layer)
        self.layerDict[layer.name] = layer

        # Use default layerConfig if None is provided
        layerConfig = self.layerConfig if layerConfig is None else layerConfig

        layer.AttachNet(self, layerConfig) # give layer a reference to the net
        # Initialize phase histories
        for phase in self.phaseConfig.keys():
            layer.phaseHist[phase] = layer.getActivity()

    def AddLayers(self, layers: list[Layer], layerConfig = None):
        # Use default layerConfig if None is provided
        layerConfig = self.layerConfig if layerConfig is None else layerConfig

        for layer in layers:
            self.AddLayer(layer, layerConfig)

    def AddConnection(self,
                      sending: Layer, # closer to source
                      receiving: Layer, # further from source
                      meshType = Mesh,
                      meshArgs = {},
                      ) -> Mesh:
        '''Adds a connection from the sending layer to the receiving layer.
        '''
        # Use default ffMeshConfig if None is provided
        size = len(receiving)
        mesh = meshType(size, sending, dtype=self.dtype, **meshArgs)
        receiving.addMesh(mesh)

        return mesh

    def AddConnections(self,
                       sendings: list[Layer], # closer to source
                       receivings: list[Layer], # further from source
                       meshType = Mesh,
                       meshArgs = {},
                       ) -> list[Mesh]:
        '''Helper function for generating multiple connections at once.
        '''
        meshes = []
        for receiving, sending in zip(receivings, sendings):
            mesh = self.AddConnection(sending, receiving, meshType, meshArgs)
            meshes.append(mesh)
        return meshes

    def AddBidirectionalConnection(self,
                      sending: Layer, # closer to source
                      receiving: Layer, # further from source
                      ffMeshType = Mesh,
                      ffMeshArgs = {},
                      fbMeshType = None,
                      fbMeshArgs = {},
                      ) -> Mesh:
        '''Adds a set of bidirectional connections from the sending layer to
            the receiving layer. The feedback mesh is assumed to be a transpose
            of the feedforward mesh.
        '''

        fbMeshType = ffMeshType if fbMeshType is None else fbMeshType

        # feedforward connection
        size = len(sending)
        ffMesh = ffMeshType(size, sending, **ffMeshArgs)
        receiving.addMesh(ffMesh)

        # feedback connection
        size = len(receiving)
        fbMesh = fbMeshType(ffMesh, receiving, **fbMeshArgs)
        sending.addMesh(fbMesh)

        return ffMesh, fbMesh

    def AddBidirectionalConnections(self,
                       sendings: list[Layer], # closer to source
                       receivings: list[Layer], # further from source
                       ffMeshType = Mesh,
                       ffMeshArgs = {},
                       fbMeshType = None,
                       fbMeshArgs = {},
                       ) -> list[Mesh]:
        '''Helper function for generating multiple bidirectional connections 
            at once.
        '''
        meshes = []
        for sending, receiving in zip(sendings, receivings):
            mesh = self.AddBidirectionalConnection(sending, receiving, ffMeshType, ffMeshArgs, fbMeshType, fbMeshArgs)
            meshes.append(mesh)
        return meshes

    def ValidateDataset(self, **dataset: dict[str, np.ndarray]):
        '''Ensures that the entered dataset is properly constructed.
        '''
        numSamples = 0
        if len(dataset) == 0:
            raise ValueError("Dataset contains no rows or columns")
        for sampleList in dataset.values():
            if numSamples == 0:
                numSamples = len(sampleList)
            elif numSamples != len(sampleList):
                raise ValueError("Ragged dataset, number of rows (samples) do not "
                                 "match across columns.")
        # Else return number of samples in dataset
        return numSamples

    def EvaluateMetrics(self, **dataset):
        '''Evaluates each metric with respect to each output layer and its
            corresponding target in the dataset. The results are stored in a
            dictionary with keys representing names of each metric.
        '''
        isFinished = False
        first = True
        for metricName, metric in self.runConfig["metrics"].items():
            result = metric(self.outputs["target"], dataset["target"])
            self.results[metricName].append(result)

            # Check the first metric for if it passes the end condition
            if first: #TODO: optimize for execution time
                first = False
                if "End" in self.runConfig:
                    if self.runConfig["End"]["isLower"]:
                        if result <= self.runConfig["End"]["threshold"]:
                            self.lrnThresh += 1
                            if self.lrnThresh >= self.runConfig["End"]["numEpochs"]:
                                isFinished = True
                        else:
                            self.lrnThresh = 0
                    else:
                        if result >= self.runConfig["End"]["threshold"]:
                            self.lrnThresh += 1
                            if self.lrnThresh >= self.runConfig["End"]["numEpochs"]:
                                isFinished = True
                        else:
                            self.lrnThresh = 0

        return isFinished

    def UpdateConductances(self):
        for layer in self.layers:
            layer.UpdateConductance()
    
    def UpdateActivity(self,):
        ## TODO: Parallelize execution for all layers
        # StepTime for each unclamped layer
        for layer in self.layerDict.values():
            if layer.name == "Input":
                continue # input layer is always clamped
            else:
                layer.StepTime(self.time,)

        # Update internal variables of input and output layer
        self.layerDict["Input"].EndStep(self.time)

    def MinusPhase(self, input: np.ndarray,):
        '''Compute a phase of execution for the neural network. A phase is a 
            set of timesteps related to some process in the network such as 
            the generation of an expectation versus observation of outcome in
            Prof. O'Reilly's error-driven local learning framework.
        '''
        # Apply clamp to the "Input" layer for the minus phase
        self.layerDict["Input"].Clamp(input, self.time, monitoring=self.monitoring)

        # Fixed 75 timesteps for Leabra minus phase
        for timeStep in range(75):
            self.UpdateConductances()
            self.UpdateActivity()

            self.time += self.DELTA_TIME

        # Record phasic history
        for layer in self.layers:
            #record phase activity at the end of each phase
            layer.phaseHist["minus"] = layer.getActivity().copy()

    def PlusPhase(self, input: np.ndarray, target: np.ndarray,):
        '''Compute a phase of execution for the neural network. A phase is a 
            set of timesteps related to some process in the network such as 
            the generation of an expectation versus observation of outcome in
            Prof. O'Reilly's error-driven local learning framework.
        '''
        # Apply clamp to the "Input" layer AND "Output" layer for the plus phase
        self.layerDict["Input"].Clamp(input, self.time, monitoring=self.monitoring)
        self.layerDict["Output"].EXTERNAL = target

        # Fixed 25 timesteps for Leabra plus phase
        for timeStep in range(25):
            self.UpdateConductances()
            self.UpdateActivity()

            self.time += self.DELTA_TIME

        # Record phasic history
        for layer in self.layers:
            #record phase activity at the end of each phase
            layer.phaseHist["plus"] = layer.getActivity().copy()

            # Execute phasic processes (ActAvg)
            layer.ActAvg.StepPhase()

    def StepTrial(self, **dataVectors):
        for layer in self.layers:
            layer.InitTrial()

        self.MinusPhase(input=dataVectors["input"],)

        # Extract the outputs of the network based on the network's inference
        self.outputs["target"].append(self.layerDict["Output"].getActivity())

        self.PlusPhase(input=dataVectors["input"], target=dataVectors["target"],)

        # Execute learning after the plus phase
        for layer in self.layers:
            layer.Learn()

    def RunEpoch(self,
                 runType: str,
                 verbosity = 1,
                 reset: bool = False,
                 shuffle: bool = False,
                 **dataset: dict[str, np.ndarray]):
        '''Runs an epoch (iteration through all samples of a dataset) using a
            specified run type mathing a key from self.runConfig.

                - verbosity: specifies how much is printed to the console
                - reset: specifies if layer activity is returned to zero each epoch
                - shuffle: determines if the dataset is shuffled each epoch
        '''
        numSamples = self.ValidateDataset(**dataset)

        # TODO use pre-allocated numpy array to speed up execution
        self.outputs = {key: [] for key in self.runConfig["outputLayers"]}

        # suffle indices if necessary
        sampleIndices = np.random.permutation(numSamples) if shuffle else range(numSamples)
        
        # TODO: find a faster way to iterate through datasets
        for sampleCount, sampleIndex in enumerate(sampleIndices):
            if verbosity > 0:
                print(f"\rEpoch: {self.epochIndex}, "
                      f"sample: ({sampleCount+1}/{numSamples}), ", end=""#"\r"
                      )

            dataVectors = {key:value[sampleIndex] for key, value in dataset.items()}
            self.StepTrial(**dataVectors)

            if reset:
                self.resetActivity()

        return numSamples

    
    def Learn(self,
              numEpochs = 50,
              verbosity = 1,
              reset: bool = True,
              shuffle: bool = True,
              batchSize = 1, # TODO: Implement batch training (average delta weights over some number of training examples)
              repeat=1, # TODO: Implement repeated sample training (train muliple times for a single input sample before moving on to the next one)
              EvaluateFirst = True,
              **dataset: dict[str, np.ndarray]) -> dict[str: list]:
        '''Training loop that runs a specified number of epochs.

                - verbosity: specifies how much is printed to the console
                - reset: specifies if layer activity is returned to zero each epoch
                - shuffle: determines if the dataset is shuffled each epoch
                - batchSize: (NOT IMPLEMENTED)
                - repeat: (NOT IMPLEMENTED)

        '''
        if EvaluateFirst:
            # Evaluate without training
            monitoring = self.monitoring
            self.monitoring = False # Temporarily pause monitoring
            if self.epochIndex == 0: # only if net has never been trained
                self.Evaluate(verbosity, reset=True, shuffle=False, **dataset)
                self.resetActivity()
            self.monitoring = monitoring # Resume normal monitoring

            self.time = 0 #TODO: allow choice to reset the timer?
            
        # Training loop
        print(f"Begin training [{self.name}]...")
        for epochIndex in range(numEpochs):
            self.epochIndex = int(EvaluateFirst) + epochIndex
            numSamples = self.RunEpoch("Learn", verbosity, reset, shuffle,
                                       **dataset)
            isFinished = self.EvaluateMetrics(**dataset)
            if verbosity > 0:
                primaryMetric = [key for key in self.runConfig["metrics"]][0]
                print(f"\rEpoch: {self.epochIndex}, "
                      f"sample: ({numSamples}/{numSamples}), "
                      f" metric[{primaryMetric}]"
                      f" = {self.results[primaryMetric][-1]}")
                
            if isFinished:
                break
                
        print(f"Finished training [{self.name}]")
        return self.results

    def Evaluate(self,
              verbosity = 1,
              reset: bool = False,
              shuffle: bool = False,
              **dataset: dict[str, np.ndarray]):

        if verbosity > 0:
            print(f"Evaluating [{self.name}] without training...")
        self.RunEpoch("Infer", verbosity, reset, shuffle, **dataset)
        self.EvaluateMetrics(**dataset)
        if verbosity > 0:
            primaryMetric = [key for key in self.runConfig["metrics"]][0]
            print(f" metric[{primaryMetric}]"
                f" = {self.results[primaryMetric][-1]:0.4f}")
            
        print("Evaluation complete.")
        return self.results
    
    def Infer(self,
              verbosity = 1,
              reset: bool = False,
              **dataset: dict[str, np.ndarray]):
        '''Applies the network to a given dataset and returns each output
        '''
        if verbosity > 0:
            print(f"Inferring [{self.name}]...")
        self.RunEpoch("Infer", verbosity, reset, shuffle= False, **dataset)
            
        if verbosity > 0:
            print("Inference complete.")
        return self.outputs

    def getWeights(self, ffOnly = True):
        weights = []
        for layer in self.layers:
            for mesh in layer.excMeshes:
                weights.append(mesh.get())
                if ffOnly:
                    break
        return weights
    
    def GetEnergy(self, synDevice = None) -> tuple[float, float]:
        neuralEnergy = 0
        meshEnergy = 0
        for layer in self.layers:
            nEn, mEn = layer.GetEnergy(synDevice=synDevice)
            neuralEnergy += nEn
            meshEnergy += mEn
        return neuralEnergy, meshEnergy
    def printActivity(self):
        for layer in self.layers:
            "\n".join(layer.printActivity())

    def resetActivity(self):
        for layer in self.layers:
            layer.resetActivity()

    def setLearningRule(self, rule, layerIndex: int = -1):
        '''Sets the learning rule for all forward meshes to 'rule'.
        '''
        if layerIndex == -1 :
            for layer in self.layers:
                layer.rule = rule
        else:
            self.layers[layerIndex].rule = rule

    def __str__(self) -> str:
        strs = []
        for layer in self.layers:
            strs.append(str(layer))

        return "\n\n".join(strs)

    def get_serial(self) -> dict:
        '''Serialize the full network to a python-native dict suitable for
           JSON saving. Numpy arrays are converted to lists.
        '''
        return {
            "name": self.name,
            "dtype": str(self.dtype),
            "time": float(self.time),
            "epochIndex": int(self.epochIndex),
            "layers": [layer.get_serial() for layer in self.layers],
        }

    def save_serial(self, filePath: str, indent: int = 4):
        '''Save full network state as JSON.'''
        with open(filePath, "w", encoding="utf-8") as handle:
            json.dump(self.get_serial(), handle, indent=indent)

    def load_serial(self, serial: dict):
        '''Load state into an already-constructed network.

        This base method assumes the user has already instantiated all layers
        and meshes in the intended topology.
        '''
        self.name = serial.get("name", self.name)
        self.time = serial.get("time", self.time)
        self.epochIndex = serial.get("epochIndex", self.epochIndex)

        # Load layers in-place by index.
        for layer, lser in zip(self.layers, serial.get("layers", [])):
            layer.load_serial(lser)

        return self

    def load_serial_file(self, filePath: str):
        '''Load full network state from JSON file into current object.'''
        with open(filePath, "r", encoding="utf-8") as handle:
            serial = json.load(handle)
        return self.load_serial(serial)