'''Defines a Net class corresponding to a group of layers interconnected with
    various meshes.
'''

# type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .layers import Layer
    from .meshes import Mesh

from collections.abc import Iterator
import math

import numpy as np

# import defaults
from .meshes import Mesh, TransposeMesh
from .metrics import RMSE
from .optimizers import Simple
from .visualize import Monitor



###<------ DEFAULT CONFIGURATIONS ------>###

ffMeshConfig_std = {
    "meshType": Mesh,
    "meshArgs": {},
}

fbMeshConfig_std = {
    "meshType": TransposeMesh,
    "meshArgs": {},
}

layerConfig_std = {
    # "DELTA_Vm" : 0.1/2.81,
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
    "ActParams": {
        "SSTau": 2,
        "STau": 2,
        "MTau": 10,
        "Tau": 10,
        "Gain": 2.5,
        "Min": 0.2,
        "LrnM": 0.1,
    },
    "optimizer": Simple,
    "optArgs": {},
    "ffMeshConfig": ffMeshConfig_std,
    "fbMeshConfig": fbMeshConfig_std,
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
        "clampLayers": {"input": 0,
                    },
    },
    "plus": {
        "numTimeSteps": 25,
        "isOutput": False,
        "clampLayers": {"input": 0,
                    "target": -1,
                    },
    },
}

runConfig_std = { # Dict of phase names
    "DELTA_TIME": 0.1,
    "metrics": {
        "RMSE": RMSE,
    },
    "outputLayers": {
        "target": -1,
    },
    "Learn": ["minus", "plus"],
    "Infer": ["minus"],
}




###<------ NET CLASSES ------>###
class Net:
    '''Base class for neural networks with Hebbian-like learning
    '''
    count = 0
    def __init__(self,
                 name = None,
                 monitoring = False,
                 runConfig = runConfig_std,
                 phaseConfig = phaseConfig_std,
                 layerConfig = layerConfig_std,
                 **kwargs):
        '''Instanstiates an ordered list of layers that will be
            applied sequentially during inference.
        '''

        self.runConfig = runConfig # Simulation-level configurations
        self.phaseConfig = phaseConfig # Dictionary of phase definitions
        self.layerConfig = layerConfig # Stereotyped layer definition
        self.monitoring = monitoring

        self.layers: list[Layer] = [] # list of layer objects
        self.layerDict: dict[str, Layer] = {} # dict of named layers

        self.name =  f"NET_{Net.count}" if name == None else name
        self.epochIndex = 0
        Net.count += 1

    def UpdateLayerLists(self):
        '''Pre-generate clamped and unclamped layer lists to speed up StepPhase
            execution.
        '''
        # TODO Figure out where to call this function (after layers have been added)
        # if len(self.layers) < len(self.phaseConfig[phaseName]["clampLayers"]):
        #     return
        for phaseName in self.phaseConfig.keys():
            layers = list(self.layers) # copy full layer list
            self.layerDict[phaseName] = {}
            self.layerDict[phaseName]["clamped"] = {}

            for dataName, layerIndex in self.phaseConfig[phaseName]["clampLayers"].items():
                if len(layers) == 0:
                    return
                self.layerDict[phaseName]["clamped"][dataName] = layers.pop(layerIndex)

            self.layerDict[phaseName]["unclamped"]: list[Layer] = layers

        self.layerDict["outputLayers"] = {}
        for dataName, index in self.runConfig["outputLayers"].items():
            self.layerDict["outputLayers"][dataName] = self.layers[index]
                
    def AddLayer(self, layer: Layer, layerConfig: dict = None):
        # index = len(self.layers)
        # size = len(layer)

        self.layers.append(layer)
        self.layerDict[layer.name] = layer

        # Use default layerConfig if None is provided
        layerConfig = self.layerConfig if layerConfig is None else layerConfig

        layer.AttachNet(self, layerConfig) # give layer a reference to the net
        # Initialize phase histories
        for phase in self.phaseConfig.keys():
            layer.phaseHist[phase] = layer.getActivity()
        
        # TODO replace monitor definition ("one function does one thing!")
        # Define monitor
        # layer.monitor = layerConfig["defMonitor"](
        #     name = self.name + ": " + layer.name,
        #     labels = ["time step", "activity"],
        #     limits=[self.runConfig["numTimeSteps"], 2],
        #     numLines=len(layer)
        # )

        self.UpdateLayerLists()

    def AddLayers(self, layers: list[Layer], layerConfig = None):
        # Use default layerConfig if None is provided
        layerConfig = self.layerConfig if layerConfig is None else layerConfig

        for layer in layers:
            self.AddLayer(layer, layerConfig)
            # size = len(layer)
            # self.layerDict[layer.name] = layer
            # layer.AttachNet(self, layerConfig) # give layer a reference to the net

            # TODO replace monitor definition ("one function does one thing!")
            # Define monitor
            # layer.monitor = layerConfig["defMonitor"](
            #     name = self.name + ": " + layer.name,
            #     labels = ["time step", "activity"],
            #     limits=[self.runConfig["numTimeSteps"], 2],
            #     numLines=len(layer)
            # )

        # self.UpdateLayerLists()

    def AddConnection(self,
                      sending: Layer, # closer to source
                      receiving: Layer, # further from source
                      meshConfig = None,
                      ):
        '''Adds a connection from the sending layer to the receiving layer.
        '''
        # Use default ffMeshConfig if None is provided
        meshConfig = self.layerConfig["ffMeshConfig"] if meshConfig is None else meshConfig
        size = len(receiving)
        meshArgs = meshConfig["meshArgs"]
        mesh = meshConfig["meshType"](size, sending, **meshArgs)
        receiving.addMesh(mesh)

    def AddConnections(self,
                       sendings: list[Layer], # closer to source
                       receivings: list[Layer], # further from source
                       meshConfig = None,
                       ):
        '''Helper function for generating multiple connections at once.
        '''
        for receiving, sending in zip(receivings, sendings):
            self.AddConnection(sending, receiving, meshConfig)

    def AddBidirectionalConnection(self,
                      sending: Layer, # closer to source
                      receiving: Layer, # further from source
                      ffMeshConfig = None,
                      fbMeshConfig = None,
                      ):
        '''Adds a set of bidirectional connections from the sending layer to
            the receiving layer. The feedback mesh is assumed to be a transpose
            of the feedforward mesh.
        '''
        # Use default ffMeshConfig if None is provided
        ffMeshConfig = self.layerConfig["ffMeshConfig"] if ffMeshConfig is None else ffMeshConfig
        fbMeshConfig = self.layerConfig["fbMeshConfig"] if fbMeshConfig is None else fbMeshConfig

        # feedforward connection
        size = len(sending)
        meshArgs = ffMeshConfig["meshArgs"]
        ffMesh = ffMeshConfig["meshType"](size, sending, **meshArgs)
        receiving.addMesh(ffMesh)

        # feedback connection
        size = len(receiving)
        meshArgs = fbMeshConfig["meshArgs"]
        fbMesh = fbMeshConfig["meshType"](ffMesh, receiving, **meshArgs)
        sending.addMesh(fbMesh)

    def AddBidirectionalConnections(self,
                       sendings: list[Layer], # closer to source
                       receivings: list[Layer], # further from source
                       meshConfig = None,
                       ):
        '''Helper function for generating multiple bidirectional connections 
            at once.
        '''
        for sending, receiving in zip(sendings, receivings):
            self.AddBidirectionalConnection(sending, receiving, meshConfig)

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
        self.results = {}
        for metricName, metric in self.runConfig["metrics"].items():
            for dataName in self.layerDict["outputLayers"]:
                self.results[metricName] = metric(self.outputs[dataName], dataset[dataName])
    
    def StepPhase(self, phaseName: str, **dataVectors):
        '''Compute a phase of execution for the neural network. A phase is a 
            set of timesteps related to some process in the network such as 
            the generation of an expectation versus observation of outcome in
            Prof. O'Reilly's error-driven local learning framework.
        '''
        numTimeSteps = self.phaseConfig[phaseName]["numTimeSteps"]
        for timeStep in range(numTimeSteps):
            ## TODO: Parallelize execution for all layers

            # Clamp layers according to phaseType
            for dataName, clampedLayer in self.layerDict[phaseName]["clamped"].items():
                    clampedLayer.Clamp(dataVectors[dataName])

            # StepTime for each unclamped layer
            for layer in self.layerDict[phaseName]["unclamped"]:
                layer.StepTime()

        # Execute phasic processes (including XCAL)
        for layer in self.layers:
            for process in layer.phaseProcesses:
                if phaseName in process.phases or "all" in process.phases:
                    process.StepPhase()
            #record phase activity at the end of each phase
            layer.phaseHist[phaseName] = layer.getActivity().copy()

    def StepTrial(self, runType: str, **dataVectors):
        for phaseName in self.runConfig[runType]:
            self.StepPhase(phaseName, **dataVectors)

            # Store layer activity for each output layer during output phases
            if self.phaseConfig[phaseName]["isOutput"]:
                for dataName, layer in self.layerDict["outputLayers"].items():
                    # TODO use pre-allocated numpy array to speed up execution
                    self.outputs[dataName].append(layer.getActivity())

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
        for sampleIndex in sampleIndices:
            if verbosity > 0:
                print(f"Epoch: {self.epochIndex}, "
                      f"sample: ({sampleIndex}/{numSamples}), ", end="\r")

            dataVectors = {key:value[sampleIndex] for key, value in dataset.items()}
            self.StepTrial(runType, **dataVectors)

            if reset : self.resetActivity()

    
    def Learn(self,
              numEpochs = 50,
              verbosity = 1,
              reset: bool = True,
              shuffle: bool = True,
              batchSize = 1, # TODO: Implement batch training (average delta weights over some number of training examples)
              repeat=1, # TODO: Implement repeated sample training (train muliple times for a single input sample before moving on to the next one)
              **dataset: dict[str, np.ndarray]):
        '''Training loop that runs a specified number of epochs.

                - verbosity: specifies how much is printed to the console
                - reset: specifies if layer activity is returned to zero each epoch
                - shuffle: determines if the dataset is shuffled each epoch
                - batchSize: (NOT IMPLEMENTED)
                - repeat: (NOT IMPLEMENTED)

        '''

        # Evaluate without training
        if self.epochIndex == 0: # only if net has never been trained
            self.Evaluate(verbosity, reset, shuffle, **dataset)
            
        # Training loop
        print(f"Begin training [{self.name}]...")
        for epochIndex in range(numEpochs):
            self.epochIndex += 1 + epochIndex
            self.RunEpoch("Learn", verbosity, reset, shuffle, **dataset)
            self.EvaluateMetrics(**dataset)
            if verbosity > 0:
                primaryMetric = [key for key in self.runConfig["metrics"]][0]
                print(f" metric[{primaryMetric}]"
                    f" = {self.outputs[primaryMetric]:0.2f}")
                
        print(f"Finished training [{self.name}]")
        return self.results

    def Evaluate(self,
              verbosity = 1,
              reset: bool = True,
              shuffle: bool = True,
              **dataset: dict[str, np.ndarray]):

        print(f"Evaluating [{self.name}] without training...")
        self.RunEpoch("Infer", verbosity, reset, shuffle, **dataset)
        self.EvaluateMetrics(**dataset)
        if verbosity > 0:
            primaryMetric = [key for key in self.runConfig["metrics"]][0]
            print(f" metric[{primaryMetric}]"
                f" = {self.results[primaryMetric]:0.2f}")
            
        print(f"Evaluatation complete.")
        return self.results
    
    def Infer(self,
              verbosity = 1,
              reset: bool = True,
              **dataset: dict[str, np.ndarray]):
        '''Applies the network to a given dataset and returns each output
        '''
        print(f"Inferring [{self.name}]...")
        self.RunEpoch("Infer", verbosity, reset, shuffle= False, **dataset)
            
        print(f"Inference complete.")
        return self.outputs

    def getWeights(self, ffOnly = True):
        weights = []
        for layer in self.layers:
            for mesh in layer.excMeshes:
                weights.append(mesh.get())
                if ffOnly: break
        return weights
    
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
    
# class RecurNet(Net):
#     '''A recurrent network with feed forward and feedback meshes
#         between each layer. Based on ideas presented in [2].
#     '''
#     def __init__(self, *args, FeedbackMesh = fbMesh, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         for index, layer in enumerate(self.layers[1:-1], 1): 
#             #skip input and output layers, add feedback matrices
#             nextLayer = self.layers[index+1]
#             layer.addMesh(FeedbackMesh(nextLayer.excMeshes[0], nextLayer))

# class FFFB(Net):
#     '''A recurrent network with feed forward and feedback meshes
#         and a trucated lateral inhibition mechanism. Based on 
#         ideas presented in [1].
#     '''
#     def __init__(self, *args, FeedbackMesh = fbMesh, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         for index, layer in enumerate(self.layers[1:-1], 1): #skip input and output layers
#             # add feedback matrices
#             nextLayer = self.layers[index+1]
#             layer.addMesh(FeedbackMesh(nextLayer.excMeshes[0], nextLayer))
#             # add FFFB inhibitory mesh
#             inhibitoryMesh = InhibMesh(layer.excMeshes[0], layer)
#             layer.addMesh(inhibitoryMesh, excitatory=False)
#         # add last layer FFFB mesh
#         layer = self.layers[-1]
#         inhibitoryMesh = InhibMesh(layer.excMeshes[0], layer)
#         layer.addMesh(inhibitoryMesh, excitatory=False)
