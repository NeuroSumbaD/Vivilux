'''Defines a Net class corresponding to a group of layers interconnected with
    various meshes.
'''

# type checking
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .layers import Layer
    from .meshes import Mesh

from collections.abc import Iterator
import math

import jax
from jax import numpy as jnp
from flax import nnx

# import defaults
from .meshes import Mesh, TransposeMesh
from .metrics import RMSE
from .visualize import Monitor
from .photonics.devices import Device



###<------ DEFAULT CONFIGURATIONS ------>###

ffMeshConfig_std = {
    "meshType": Mesh,
    "meshArgs": {},
}

fbMeshConfig_std = {
    "meshType": TransposeMesh,
    "meshArgs": {"RelScale": 0.2,
                 },
}

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
        "isLearn": False,
        "inputClamped": {"input": 0,
                         },
        "targetClamped": {
                          },
    },
    "plus": {
        "numTimeSteps": 25,
        "isOutput": False,
        "isLearn": True,
        "inputClamped": {"input": 0,
                         },
        "targetClamped": {"target": -1,
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
    "Infer": ["minus"],
    "End": {
        "threshold": 0,
        "isLower": True,
        "numEpochs": 5,
    }
}




###<------ NET CLASSES ------>###
class Net(nnx.Module):
    '''Base class for neural networks with Hebbian-like learning
    '''
    count = 0
    def __init__(self,
                 name = None,
                 monitoring = False,
                 runConfig = runConfig_std,
                 phaseConfig = phaseConfig_std,
                 layerConfig = layerConfig_std,
                 dtype = jnp.float32, # TODO: figure out float64 support
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 **kwargs):
        '''Instanstiates an ordered list of layers that will be
            applied sequentially during inference.
        '''

        self.runConfig = runConfig # Simulation-level configurations
        self.phaseConfig: dict[str, dict] = phaseConfig # Dictionary of phase definitions
        self.layerConfig: dict = layerConfig # Stereotyped layer definition
        self.monitoring = monitoring
        self.dtype = dtype
        self.rngs = rngs

        # For time keeping
        self.DELTA_TIME = runConfig["DELTA_TIME"]
        self.time = 0

        self.layers: list[Layer] = [] # list of layer objects
        self.layerNames: dict[str, Layer] = {} # dict of named layers
        self.layerTags: dict[str, dict[str, Layer]] = {} # collections of layers for phase and output tagging

        self.results: dict[str, list[jnp.ndarray]] = {metric: [] for metric in self.runConfig["metrics"]}
        self.outputs: dict[str, list[jnp.ndarray]] = {key: [] for key in self.runConfig["outputLayers"]}

        self.name =  f"NET_{Net.count}" if name == None else name
        self.epochIndex = 0
        Net.count += 1

        # For early stopping learning process
        self.lrnThresh = 0

    def PreallocateResultDict(self):
        '''Pre-allocate a dict to store the results
        '''
    
    def UpdateLayerLists(self):
        '''Pre-generate clamped and unclamped layer lists to speed up StepPhase
            execution.
        '''
        self.layerNames = {layer.name: layer for layer in self.layers}
        layerTags: dict[str, dict[str, Layer]] = {} # collections of layers for phase and output tagging

        for phaseName in self.phaseConfig.keys():
            layers = list(self.layers) # copy full layer list
            layerTags[phaseName] = {}
            layerTags[phaseName]["inputClamped"] = {}
            layerTags[phaseName]["targetClamped"] = {}

            for dataName, layerIndex in self.phaseConfig[phaseName]["inputClamped"].items():
                if len(layers) == 0:
                    return
                layerTags[phaseName]["inputClamped"][dataName] = layers.pop(layerIndex)

            for dataName, layerIndex in self.phaseConfig[phaseName]["targetClamped"].items():
                if len(layers) == 0:
                    return
                layerTags[phaseName]["targetClamped"][dataName] = layers.pop(layerIndex)

            layerTags[phaseName]["unclamped"] = layers

        layerTags["outputLayers"] = {}
        for dataName, index in self.runConfig["outputLayers"].items():
            layerTags["outputLayers"][dataName] = self.layers[index]
        
        self.layerTags = layerTags
                
    def AddLayer(self, layer: Layer, layerConfig: dict = None):
        # index = len(self.layers)
        # size = len(layer)

        if layer.dtype != self.dtype:
            layer.SetDtype(self.dtype)

        self.layers.append(layer)
        self.layerTags[layer.name] = layer

        # Use default layerConfig if None is provided
        layerConfig = self.layerConfig if layerConfig is None else layerConfig

        layer.AttachNet(self, layerConfig, self.rngs) # give layer a reference to the net
        # Initialize phase histories
        # for phase in self.phaseConfig.keys():
        #     layer.phaseHist[phase] = layer.getActivity()
        layer.phaseHist = nnx.Variable(jnp.zeros((len(self.phaseConfig.keys()), len(layer))))
        layer.ActAvgPhaseIndex.value = list(self.phaseConfig.keys()).index("plus") # default phase for ActAvg
        self.UpdateLayerLists()

    def AddLayers(self, layers: list[Layer], layerConfig = None):
        # Use default layerConfig if None is provided
        layerConfig = self.layerConfig if layerConfig is None else layerConfig

        for layer in layers:
            self.AddLayer(layer, layerConfig)

    def AddConnection(self,
                      sending: Layer, # closer to source
                      receiving: Layer, # further from source
                      meshConfig = None,
                      device: Device = None,
                      ) -> Mesh:
        '''Adds a connection from the sending layer to the receiving layer.
        '''
        # Use default ffMeshConfig if None is provided
        meshConfig = self.layerConfig["ffMeshConfig"] if meshConfig is None else meshConfig
        size = len(receiving)
        meshArgs = meshConfig["meshArgs"]
        mesh: Mesh = meshConfig["meshType"](size,
                                            sending,
                                            self.rngs,
                                            dtype=self.dtype,
                                            **meshArgs)
        receiving.addMesh(mesh)

        if device is not None:
            mesh.AttachDevice(device)

        return mesh

    def AddConnections(self,
                       sendings: list[Layer], # closer to source
                       receivings: list[Layer], # further from source
                       meshConfig = None,
                       device: Device = None,
                       ) -> list[Mesh]:
        '''Helper function for generating multiple connections at once.
        '''
        meshes = []
        for receiving, sending in zip(receivings, sendings):
            mesh = self.AddConnection(sending, receiving, meshConfig, device)
            meshes.append(mesh)
        return meshes

    def AddBidirectionalConnection(self,
                      sending: Layer, # closer to source
                      receiving: Layer, # further from source
                      ffMeshConfig = None,
                      fbMeshConfig = None,
                      ) -> Mesh:
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
        ffMesh = ffMeshConfig["meshType"](size,
                                          sending,
                                          self.rngs,
                                          **meshArgs)
        receiving.addMesh(ffMesh)

        # feedback connection
        size = len(receiving)
        meshArgs = fbMeshConfig["meshArgs"]
        fbMesh = fbMeshConfig["meshType"](ffMesh,
                                          receiving,
                                          self.rngs,
                                          **meshArgs)
        sending.addMesh(fbMesh)

        return ffMesh, fbMesh

    def AddBidirectionalConnections(self,
                       sendings: list[Layer], # closer to source
                       receivings: list[Layer], # further from source
                       meshConfig = None,
                       ) -> list[Mesh]:
        '''Helper function for generating multiple bidirectional connections 
            at once.
        '''
        meshes = []
        for sending, receiving in zip(sendings, receivings):
            mesh = self.AddBidirectionalConnection(sending, receiving, meshConfig)
            meshes.append(mesh)
        return meshes

    def ValidateDataset(self, **dataset: dict[str, jnp.ndarray]):
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
            for dataName in self.layerTags["outputLayers"]:
                result = metric(self.outputs[dataName], dataset[dataName])
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

    def ClampLayers(self, phaseName: str, debugData = None, **dataVectors):
        # Clamp layers according to phaseType
        ## TODO: Change clamp to execute outside time loop, unclamp after, & update important internal variables
        inputClamped: dict[str, Layer] = self.layerTags[phaseName]["inputClamped"]
        for dataName, clampedLayer in inputClamped.items():
            clampedLayer.Clamp(dataVectors[dataName], self.time, debugData=debugData)

        targetClamped: dict[str, Layer] = self.layerTags[phaseName]["targetClamped"]
        for dataName, clampedLayer in targetClamped.items():
            clampedLayer.EXTERNAL = dataVectors[dataName]
    
    
    def UpdateActivity(self, phaseName: str, debugData={}, **dataVectors):
        ## TODO: Parallelize execution for all layers
        # StepTime for each unclamped layer
        for layer in self.layerTags[phaseName]["unclamped"]:
            # debugData = dataVectors["debugData"] if "debugData" in dataVectors else None
            layer.StepTime(self.time, debugData=debugData)

        # Update internal variables of clamped layers
        first = True ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
        for layer in self.layerTags[phaseName]["clamped"].values():
            # debugData = dataVectors["debugData"] if "debugData" in dataVectors else None
            # layer.UpdateConductance()
            if first: ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
                layer.EndStep(self.time, debugData=debugData)
                first = False
            else: ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
                layer.StepTime(self.time, debugData=debugData)
    
    def StepPhase(self, phaseName: str, debugData = {}, **dataVectors):
        '''Compute a phase of execution for the neural network. A phase is a 
            set of timesteps related to some process in the network such as 
            the generation of an expectation versus observation of outcome in
            Prof. O'Reilly's error-driven local learning framework.
        '''
        numTimeSteps = self.phaseConfig[phaseName]["numTimeSteps"]
        
        self.ClampLayers(phaseName, **dataVectors)

        for timeStep in range(numTimeSteps):
            self.UpdateConductances()
            self.UpdateActivity(phaseName, debugData=debugData, **dataVectors)

            self.time += self.DELTA_TIME

        # Execute phasic processes (including XCAL)
        for layer in self.layers:
            #record phase activity at the end of each phase
            layer.phaseHist[phaseName] = layer.getActivity().copy()

            #Execute phasic processes (including XCAL)
            for process in layer.phaseProcesses:
                if phaseName in process.phases or "all" in process.phases:
                    process.StepPhase()

    def StepTrial(self, runType: str, debugData = {}, **dataVectors):
        Train = runType=="Learn"
        for layer in self.layers:
            layer.InitTrial(Train)
            
        for phaseName in self.runConfig[runType]:
            self.StepPhase(phaseName, debugData=debugData, **dataVectors)

            # Store layer activity for each output layer during output phases
            if self.phaseConfig[phaseName]["isOutput"]:
                for dataName, layer in self.layerTags["outputLayers"].items():
                    # TODO use pre-allocated numpy array to speed up execution
                    self.outputs[dataName].append(layer.getActivity())

            if self.phaseConfig[phaseName]["isLearn"] and Train:
                for layer in self.layers:
                    dwtLog = debugData["dwtLog"] if "dwtLog" in debugData else None
                    layer.Learn(dwtLog=dwtLog)

    def UpdateConductances_jit(self):
        for layer in self.layers:
            layer.UpdateConductance()

    # @nnx.jit(static_argnames=["inputClamped", "targetClamped"])
    def ClampLayers_jit(self,
                        inputClamped: list[str],
                        targetClamped: list[str],
                        **dataVectors):
        # Clamp layers according to phaseType
        for dataName in inputClamped:
            clampedLayer = self.layerNames[dataName]
            clampedLayer.Clamp(dataVectors[dataName])

        for dataName in targetClamped:
            clampedLayer = self.layerNames[dataName]
            # TODO: this syntax may not work with jit, may need to be changed
            clampedLayer.hasExternal.value = True
            clampedLayer.EXTERNAL.value = dataVectors[dataName]
    
    # @nnx.jit(static_argnames=["unclamped", "input_clamped", "target_clamped"])
    def UpdateActivity_jit(self,
                           unclamped: list[str],
                           input_clamped: list[str],
                           target_clamped: list[str],
                           **dataVectors):
        ## TODO: Parallelize execution for all layers
        # StepTime for each unclamped layer
        for layer_name in unclamped:
            layer: Layer = self.layerNames[layer_name]
            layer.StepTime_jit()

        # Update internal variables of clamped layers
        for layer_name in input_clamped:
            layer: Layer = self.layerNames[layer_name]
            layer.EndStep_jit()

        # NOTE: it is important target clamped layers are updated last,
        # but I am not sure exactly why.
        for layer_name in target_clamped:
            layer: Layer = self.layerNames[layer_name]
            layer.StepTime_jit()


    # @nnx.jit(static_argnames=["numPhases",
    #                           "phase_list",
    #                           "unclamped",
    #                           "inputClamped",
    #                           "targetClamped",
    #                           "output_names",
    #                           "train"])
    def StepTrial_jit(self,
                      num_phases: int,
                      phase_list: tuple[str],
                      unclamped: tuple[tuple[str]],
                      inputClamped: tuple[tuple[str]],
                      targetClamped: tuple[tuple[str]],
                      output_names: tuple[str],
                      train: bool,
                      **dataVectors: jnp.ndarray):
        for layer in self.layers:
            layer.InitTrial(train)

        # Pre-allocate output storage
        trial_outputs = {}

        for phaseIndex, phaseName in enumerate(phase_list):
            self.StepPhase_jit(phaseIndex,
                               phaseName,
                               unclamped[phaseIndex],
                               inputClamped[phaseIndex],
                               targetClamped[phaseIndex],
                               **dataVectors)

            # Store layer activity for each output layer during output phases
            if self.phaseConfig[phaseName]["isOutput"]:
                for name in output_names:
                    trial_outputs[name] = self.layerNames[name].getActivity()

            if self.phaseConfig[phaseName]["isLearn"] and train:
                for layer in self.layers:
                    layer.Learn_jit()

        return trial_outputs

    # @nnx.jit(static_argnames=["phaseName", "unclamped", "inputClamped", "targetClamped"])
    def StepPhase_jit(self,
                      phaseIndex: int,
                      phaseName: str,
                      unclamped: tuple[str],
                      inputClamped: tuple[str],
                      targetClamped: tuple[str],
                      **dataVectors):
        '''Compute a phase of execution for the neural network. A phase is a 
            set of timesteps related to some process in the network such as 
            the generation of an expectation versus observation of outcome in
            Prof. O'Reilly's error-driven local learning framework.
        '''
        numTimeSteps = self.phaseConfig[phaseName]["numTimeSteps"]

        self.ClampLayers_jit(inputClamped,
                             targetClamped,
                             **dataVectors)

        for timeStep in range(numTimeSteps):
            self.UpdateConductances_jit()
            self.UpdateActivity_jit(unclamped,
                                    inputClamped,
                                    targetClamped,
                                    **dataVectors)

            self.time += self.DELTA_TIME

        # TODO: Update refactor XCAL to update state for each layer
        for layer in self.layers:
            layer.UpdatePhaseHist(phaseIndex)
            layer.UpdateActAvg(phaseIndex)
            # layer.UpdateXCAL_jit(phaseIndex)

    def RunEpoch(self,
                 runType: str,
                 verbosity = 1,
                 reset: bool = False,
                 shuffle: bool = False,
                 debugData = {},
                 **dataset: dict[str, jnp.ndarray]):
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
        sampleIndices = jax.random.permutation(self.rngs["Params"](), numSamples) if shuffle else range(numSamples)
        
        # TODO: find a faster way to iterate through datasets
        for sampleCount, sampleIndex in enumerate(sampleIndices):
            if verbosity > 0:
                print(f"\rEpoch: {self.epochIndex}, "
                      f"sample: ({sampleCount+1}/{numSamples}), ", end=""#"\r"
                      )

            dataVectors = {key:value[sampleIndex] for key, value in dataset.items()}

            if len(debugData) == 0:
                # The jitted path needs to compile a different StepTrial based
                # on the runType. Also compiles a different StepPhase_jit
                # based on the phase list, inputClamped, and targetClamped.
                train = runType == "Learn"
                phase_list = tuple(self.runConfig[runType])
                numPhases = int(len(phase_list))
                unclamped: tuple[tuple[str]] = tuple(
                    tuple(layer.name for layer in self.layerTags[phase]["unclamped"])
                    for phase in self.runConfig[runType]
                )
                inputClamped: tuple[tuple[str]] = tuple(
                    tuple(layerName for layerName in self.layerTags[phase]["inputClamped"].keys())
                    for phase in self.runConfig[runType]
                )
                targetClamped: tuple[tuple[str]] = tuple(
                    tuple(layerName for layerName in self.layerTags[phase]["targetClamped"].keys())
                    for phase in self.runConfig[runType]
                )
                output_names = tuple(self.layerTags["outputLayers"].keys())
                trial_outputs = self.StepTrial_jit(numPhases,
                                                   phase_list,
                                                   unclamped,
                                                   inputClamped,
                                                   targetClamped,
                                                   output_names,
                                                   train, 
                                                   **dataVectors)

                # dynamically add outputs to the output dict outside of jit context
                for name in trial_outputs:
                    self.outputs[name].append(trial_outputs[name])
            else: # debugging mode cannot use jit
                self.StepTrial(runType, debugData=debugData, **dataVectors)

            if reset : self.resetActivity()

        return numSamples

    
    def Learn(self,
              numEpochs = 50,
              verbosity = 1,
              reset: bool = True,
              shuffle: bool = True,
              batchSize = 1, # TODO: Implement batch training (average delta weights over some number of training examples)
              repeat=1, # TODO: Implement repeated sample training (train muliple times for a single input sample before moving on to the next one)
              EvaluateFirst = True,
              debugData = {},
              **dataset: dict[str, jnp.ndarray]) -> dict[str: list]:
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
                                       debugData=debugData, **dataset)
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
              reset: bool = True,
              shuffle: bool = True,
              **dataset: dict[str, jnp.ndarray]):

        if verbosity > 0: print(f"Evaluating [{self.name}] without training...")
        self.RunEpoch("Infer", verbosity, reset, shuffle, **dataset)
        self.EvaluateMetrics(**dataset)
        if verbosity > 0:
            primaryMetric = [key for key in self.runConfig["metrics"]][0]
            print(f" metric[{primaryMetric}]"
                f" = {self.results[primaryMetric][-1]:0.4f}")
            
        print(f"Evaluation complete.")
        return self.results
    
    def Infer(self,
              verbosity = 1,
              reset: bool = False,
              **dataset: dict[str, jnp.ndarray]):
        '''Applies the network to a given dataset and returns each output
        '''
        if verbosity > 0: print(f"Inferring [{self.name}]...")
        self.RunEpoch("Infer", verbosity, reset, shuffle= False, **dataset)
            
        if verbosity > 0: print(f"Inference complete.")
        return self.outputs

    def getWeights(self, ffOnly = True):
        weights = []
        for layer in self.layers:
            for mesh in layer.excMeshes:
                weights.append(mesh.get())
                if ffOnly: break
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