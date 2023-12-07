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
from .meshes import fbMesh, InhibMesh
from .metrics import RMSE
from .optimizers import Simple
from .visualize import Monitor

class Net:
    '''Base class for neural networks with Hebbian-like learning
    '''
    count = 0
    def __init__(self, layers: list[Layer], meshType: Mesh,
                 metric = RMSE, name = None,
                 optimizer = Simple,
                 optArgs = {},
                 meshArgs = {},
                 numTimeSteps = 50,
                 monitoring = False,
                 defMonitor = Monitor,
                 **kwargs):
        '''Instanstiates an ordered list of layers that will be
            applied sequentially during inference.
        '''
        self.DELTA_TIME = 0.1
        self.numTimeSteps = numTimeSteps
        self.monitoring = monitoring
        self.defMonitor = defMonitor

        self.name =  f"NET_{Net.count}" if name == None else name
        Net.count += 1

        # TODO: allow different mesh types between layers
        self.layers = layers
        self.metrics = [metric] if not isinstance(metric, list) else metric

        if monitoring: 
            index, layer = 0, self.layers[0]
            layer.monitor = self.defMonitor(name = self.name + ": " + layer.name,
                                    labels = ["time step", "activity"],
                                    limits=[numTimeSteps, 2],
                                    numLines=len(layer))

        for index, layer in enumerate(self.layers[1:], 1):
            size = len(layer)
            layer.addMesh(meshType(size, self.layers[index-1],
                                   **meshArgs))
            layer.optimizer = optimizer(**optArgs)
            if monitoring:
                layer.monitor = self.defMonitor(name = self.name + ": " + layer.name,
                                    labels = ["time step", "activity"],
                                    limits=[numTimeSteps, 2],
                                    numLines=len(layer))

    def Predict(self, data):
        '''Inference method called 'prediction' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        #Clamp input layer, set minus phase history
        self.layers[0].Clamp(data, monitoring=self.monitoring)
        self.layers[0].phaseHist["minus"][:] = self.layers[0].getActivity()

        for layer in self.layers[1:-1]:
            layer.Predict(monitoring=self.monitoring)
            
        output = self.layers[-1].Predict(monitoring=self.monitoring)
        
        
        return output

    def Observe(self, inData, outData):
        '''Training method called 'observe' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        #Clamp input layer, set minus phase history
        inLayer = self.layers[0]
        inLayer.Clamp(inData, monitoring=self.monitoring)
        inActivity = inLayer.getActivity()
        inLayer.phaseHist["plus"][:] = inActivity
        deltaPAvg = np.mean(inActivity) - inLayer.ActPAvg
        inLayer.ActPAvg += self.DELTA_TIME/50*(deltaPAvg) #For updating Gscale
        inLayer.snapshot["deltaPAvg"] = deltaPAvg

        #Clamp output layer, set minus phase history
        outLayer = self.layers[-1]
        outLayer.Clamp(outData, monitoring=self.monitoring)
        outActivity = outLayer.getActivity()
        outLayer.phaseHist["plus"][:] = outActivity
        deltaPAvg = np.mean(outActivity) - outLayer.ActPAvg
        outLayer.ActPAvg += self.DELTA_TIME/50*(deltaPAvg) #For updating Gscale
        outLayer.snapshot["deltaPAvg"] = deltaPAvg
        
        for layer in self.layers[1:-1]:
            layer.Observe(monitoring=self.monitoring)

        return None # observations know the outcome

    def Infer(self, inData, numTimeSteps=50, reset=False):
        outputData = np.zeros((len(inData), len(self.layers[-1])))
        index = 0
        for inDatum in inData:
            if reset: self.resetActivity()
            for time in range(numTimeSteps):
                result = self.Predict(inDatum)
            outputData[index][:] = result
            index += 1
        return outputData

    
    def Learn(self, inData: np.ndarray, outData: np.ndarray,
              numTimeSteps = None, numEpochs=50, batchSize = 1, repeat=1,
              verbose = False, reset = False, shuffle = True):
        '''Control loop for learning based on GeneRec-like algorithms.
                inData      : input data
                outData     : 
                verbose     : if True, prints net each iteration
                reset       : if True, resets activity between each input sample
        '''
        # allow update of numTimeSteps
        if numTimeSteps is not None:
            self.numTimeSteps = numTimeSteps

        # isolate input and output data
        # inData = deepcopy(inData)
        # outData = deepcopy(outData)

        results = [np.zeros(numEpochs+1) for metric in self.metrics]
        index = 0
        numSamples = len(inData)

        inData = inData.reshape(-1,1) if len(inData.shape) == 1 else inData
        outData = outData.reshape(-1,1) if len(outData.shape) == 1 else outData

        # Temporarily pause monitoring
        monitoring = self.monitoring
        self.monitoring = False
        # Evaluate without training
        print(f"Progress [{self.name}]:")
        print(f"Epoch: 0, sample: ({index}/{numSamples}), metric[{self.metrics[0].__name__}] = {results[0][0]:0.2f}  ", end="\r")
        firstResult = self.Evaluate(inData, outData, self.numTimeSteps, reset)
        for indexMetric, metric in enumerate(self.metrics):
            results[indexMetric][0] = firstResult[indexMetric]
        print(f"Epoch: 0, sample: ({index}/{numSamples}), metric[{self.metrics[0].__name__}] = {results[0][0]:0.2f}  ", end="\r")
        # Unpause monitoring
        self.monitoring = monitoring

        epochResults = np.zeros((len(outData), len(self.layers[-1])))
        # epochResults = np.zeros((len(outData), repeat, len(self.layers[-1])))
        # add mechanism for repetitions

        # batch mode
        if batchSize > 1:
            for layer in self.layers:
                layer.batchMode = True
        
        for epoch in range(numEpochs):
            if shuffle:
                permute = np.random.permutation(len(inData))
                inData, outData = inData[permute], outData[permute]
            index=0
            if batchSize > 1:
                batchInData = [inData[batchSize*i:batchSize*(i+1)] for i in range(math.ceil(len(inData)/batchSize))]
                batchOutData = [outData[batchSize*i:batchSize*(i+1)] for i in range(math.ceil(len(outData)/batchSize))]
            else:
                batchInData = inData.reshape(1,*inData.shape)
                batchOutData = outData.reshape(1,*outData.shape)
            # iterate through data and time
            for inBatch, outBatch in zip(batchInData, batchOutData):
                for inDatum, outDatum in zip(inBatch, outBatch):
                    for iteration in range(repeat):
                        if reset: self.resetActivity()
                        # TODO: MAKE ACTIVATIONS CONTINUOUS
                        ### Data should instead be recorded and labeled at the end of each phase
                        for time in range(self.numTimeSteps):
                            lastResult = self.Predict(inDatum)
                        epochResults[index][:] = lastResult
                        index += 1
                        for time in range(self.numTimeSteps):
                            self.Observe(inDatum, outDatum)
                        # update meshes
                        for layer in self.layers:
                            layer.Learn()
                    print(f"Epoch: ({epoch}/{numEpochs}), sample: ({index}/{numSamples}), metric[{self.metrics[0].__name__}] = {results[0][epoch]:0.4f}  ", end="\r")
            if batchSize > 1: # batched mode
                for layer in self.layers:
                    layer.Learn(batchComplete=True)
            # evaluate metric
            # Record multiple metrics
            for indexMetric, metric in enumerate(self.metrics):
                results[indexMetric][epoch+1] = metric(epochResults, outData)
            print(f"Epoch: ({epoch}/{numEpochs}), sample: ({index}/{numSamples}), metric[{self.metrics[0].__name__}] = {results[0][epoch+1]:0.4f}  ", end="\r")
            if verbose: print(self)
        print("\n")
        # Unpack result if there is only one metric (for backward compatibility)
        if len(results) == 1:
            return results[0]
        return results
    
    def Evaluate(self, inData, outData, numTimeSteps=25, reset=False):
        results = self.Infer(inData, numTimeSteps, reset)

        return [metric(results, outData) for metric in self.metrics]

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
    

class RecurNet(Net):
    '''A recurrent network with feed forward and feedback meshes
        between each layer. Based on ideas presented in [2].
    '''
    def __init__(self, *args, FeedbackMesh = fbMesh, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for index, layer in enumerate(self.layers[1:-1], 1): 
            #skip input and output layers, add feedback matrices
            nextLayer = self.layers[index+1]
            layer.addMesh(FeedbackMesh(nextLayer.excMeshes[0], nextLayer))

class FFFB(Net):
    '''A recurrent network with feed forward and feedback meshes
        and a trucated lateral inhibition mechanism. Based on 
        ideas presented in [1].
    '''
    def __init__(self, *args, FeedbackMesh = fbMesh, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for index, layer in enumerate(self.layers[1:-1], 1): #skip input and output layers
            # add feedback matrices
            nextLayer = self.layers[index+1]
            layer.addMesh(FeedbackMesh(nextLayer.excMeshes[0], nextLayer))
            # add FFFB inhibitory mesh
            inhibitoryMesh = InhibMesh(layer.excMeshes[0], layer)
            layer.addMesh(inhibitoryMesh, excitatory=False)
        # add last layer FFFB mesh
        layer = self.layers[-1]
        inhibitoryMesh = InhibMesh(layer.excMeshes[0], layer)
        layer.addMesh(inhibitoryMesh, excitatory=False)