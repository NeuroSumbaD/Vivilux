'''
A library for Hebbian-like learning implementations on MZI meshes based on the
work of O'Reilly et al. [1] in computation.

REFERENCES:
[1] O'Reilly, R. C., Munakata, Y., Frank, M. J., Hazy, T. E., and
    Contributors (2012). Computational Cognitive Neuroscience. Wiki Book,
    4th Edition (2020). URL: https://CompCogNeuro.org

[2] R. C. O'Reilly, “Biologically Plausible Error-Driven Learning Using
    Local Activation Differences: The Generalized Recirculation Algorithm,”
    Neural Comput., vol. 8, no. 5, pp. 895-938, Jul. 1996, 
    doi: 10.1162/neco.1996.8.5.895.
'''

from __future__ import annotations
from collections.abc import Iterator

import numpy as np
np.random.seed(seed=0)

# import defaults
from .activations import Sigmoid
from .metrics import RMSE
from .learningRules import CHL

# library constants
DELTA_TIME = 0.1

class Net:
    '''Base class for neural networks with Hebbian-like learning
    '''
    count = 0
    def __init__(self, layers: list[Layer], meshType, metric = RMSE, learningRate = 0.1, name = None):
        '''Instanstiates an ordered list of layers that will be
            applied sequentially during inference.
        '''
        self.name =  f"NET_{Layer.count}" if name == None else name
        Net.count += 1

        # TODO: allow different mesh types between layers
        self.layers = layers
        self.metric = metric

        for index, layer in enumerate(self.layers[1:], 1):
            size = len(layer)
            layer.addMesh(meshType(size, self.layers[index-1], learningRate))

    def Predict(self, data):
        '''Inference method called 'prediction' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        # outputs = []
        assert np.any(data<1), f"PREDICT ERROR: INPUT {data} GREATER THAN 1"
        self.layers[0].Clamp(data)

        for layer in self.layers[1:-1]:
            layer.Predict()
            assert np.any(layer.outAct<1), f"PREDICT ERROR: EXPLODING ACTIVATION IN {self.name},{layer.name}"

        output = self.layers[-1].Predict()
        
        return output

    def Observe(self, inData, outData):
        '''Training method called 'observe' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        assert np.any(inData<1), f"OBSERVE ERROR: INPUT {inData} GREATER THAN 1"
        assert np.any(outData<1), f"OBSERVE ERROR: OUTPUT {outData} GREATER THAN 1"
        self.layers[0].Clamp(inData)
        self.layers[-1].Clamp(outData)
        for layer in self.layers[1:-1]:
            layer.Observe()
            assert np.any(layer.outAct<1), f"OBSERVE ERROR: EXPLODING ACTIVATION IN {self.name},{layer.name}"
        # self.layers[-1].ClampObs(outData)

        return None # observations know the outcome

    def Infer(self, inData, numTimeSteps=25):
        outputData = np.zeros(inData.shape)
        index = 0
        for inDatum in inData:
            for time in range(numTimeSteps):
                result = self.Predict(inDatum)
            outputData[index][:] = result
            index += 1
        return outputData

    
    def Learn(self, inData, outData,
              numTimeSteps=50, numEpochs=50,
              verbose = False, reset = False):
        '''Control loop for learning based on GeneRec-like algorithms.
                inData      : input data
                outData     : 
                verbose     : if True, prints net each iteration
                reset       : if True, resets activity between each input sample
        '''
        results = np.zeros(numEpochs+1)
        results[0] = self.Evaluate(inData, outData, numTimeSteps)
        epochResults = np.zeros((len(outData), len(self.layers[-1])))
        for epoch in range(numEpochs):
            # iterate through data and time
            index=0
            for inDatum, outDatum in zip(inData, outData):
                if reset: self.resetActivity()
                # TODO: MAKE ACTIVATIONS CONTINUOUS
                ### Data should instead be recorded and labeled at the end of each phase
                for time in range(numTimeSteps):
                    lastResult = self.Predict(inDatum)
                epochResults[index][:] = lastResult
                index += 1
                for time in range(numTimeSteps):
                    self.Observe(inDatum, outDatum)
                # update meshes
                for layer in self.layers:
                    layer.Learn()
            # evaluate metric
            results[epoch+1] = self.metric(epochResults, outData)
            if verbose: print(self)
        
        return results
    
    def Evaluate(self, inData, outData, numTimeSteps=25):
        results = self.Infer(inData, numTimeSteps)
        return self.metric(results, outData)

    def getWeights(self, ffOnly):
        weights = []
        for layer in self.layers:
            for mesh in layer.meshes:
                weights.append(mesh.get())
                if ffOnly: break
        return weights
    
    def getActivity(self):
        for layer in self.layers:
            "\n".join(layer.getActivity())

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

class Mesh:
    '''Base class for meshes of synaptic elements.
    '''
    count = 0
    def __init__(self, size: int, inLayer: Layer, learningRate=0.5):
        self.size = size if size > len(inLayer) else len(inLayer)
        self.matrix = np.eye(self.size)
        self.inLayer = inLayer
        self.rate = learningRate


        self.name = f"MESH_{Mesh.count}"
        Mesh.count += 1

    def set(self, matrix):
        self.matrix = matrix

    def get(self):
        return self.matrix
    
    def getInput(self):
        return self.inLayer.outAct

    def apply(self):
        try:
            data = self.getInput()
            return self.get() @ data
        except ValueError as ve:
            print(f"Attempted to apply {data} (shape: {data.shape}) to mesh "
                  f"of dimension: {self.matrix}")
            
    def applyTo(self, data):
        try:
            return self.get() @ data
        except ValueError as ve:
            print(f"Attempted to apply {data} (shape: {data.shape}) to mesh "
                  f"of dimension: {self.matrix}")

    def Update(self, delta):
        self.matrix += self.rate*delta

    def __len__(self):
        return self.size

    def __str__(self):
        return f"\n\t\t{self.name.upper()} ({self.size} <={self.inLayer.name}) = {self.get()}"

class fbMesh(Mesh):
    '''A class for feedback meshes based on the transpose of another mesh.
    '''
    def __init__(self, mesh: Mesh, inLayer: Layer) -> None:
        super().__init__(mesh.size, inLayer)
        self.name = "TRANSPOSE_" + mesh.name
        self.mesh = mesh

    def set(self):
        raise Exception("Feedback mesh has no 'set' method.")

    def get(self):
        return self.mesh.get().T
    
    def getInput(self):
        return self.mesh.inLayer.outAct

    def Update(self, delta):
        return None

class Layer:
    '''Base class for a layer that includes input matrices and activation
        function pairings. Each layer retains a seperate state for predict
        and observe phases, along with a list of input meshes applied to
        incoming data.
    '''
    count = 0
    def __init__(self, length, activation=Sigmoid, learningRule=CHL,
                 isInput = False, freeze = False, name = None):
        self.inAct = np.zeros(length) # linearly integrated dendritic inputs
        self.outAct = np.zeros(length) # axonal outputs after nonlinearity
        self.phaseHist = {"minus": np.zeros(length),
                          "plus": np.zeros(length)
                          }
        
        self.act = activation
        self.rule = learningRule
        self.meshes: list[Mesh] = [] #empty initial mesh list

        self.isInput = isInput
        self.freeze = False
        self.name =  f"LAYER_{Layer.count}" if name == None else name
        if isInput: self.name = "INPUT_" + self.name
        Layer.count += 1

    def Freeze(self):
        self.freeze = True

    def Unfreeze(self):
        self.freeze = False
    
    def addMesh(self, mesh):
        self.meshes.append(mesh)

    def Predict(self):
        self.inAct -= DELTA_TIME*self.inAct
        for mesh in self.meshes:
            self.inAct += DELTA_TIME * mesh.apply()[:len(self)]**2
        self.outAct = self.act(self.inAct)
        self.phaseHist["minus"] = self.outAct.copy()
        return self.outAct

    def Observe(self):
        self.inAct -= DELTA_TIME * self.inAct
        for mesh in self.meshes:
            self.inAct += DELTA_TIME * mesh.apply()[:len(self)]**2
        self.outAct = self.act(self.inAct)
        self.phaseHist["plus"] = self.outAct.copy()
        return self.outAct

    def Clamp(self, data):
        self.inAct = data[:len(self)]
        self.outAct = data[:len(self)]

    def Learn(self):
        if self.isInput or self.freeze: return
        # TODO: Allow multiple meshes to learn, skip fb meshes
        inLayer = self.meshes[0].inLayer # assume first mesh as input
        delta = self.rule(inLayer, self)
        self.meshes[0].Update(delta)

    def getActivity(self):
        return [self.inAct, self.outAct]
    
    def resetActivity(self):
        '''Resets all activation traces to zero vectors.'''
        length = len(self)
        self.inAct = np.zeros(length)
        self.outAct = np.zeros(length)
        
    def __len__(self):
        return len(self.inAct)

    def __str__(self) -> str:
        layStr = f"{self.name} ({len(self)}): \n\tActivation = {self.act}\n\tLearning"
        layStr += f"Rule = {self.rule}"
        layStr += f"\n\tMeshes: " + "\n".join([str(mesh) for mesh in self.meshes])
        layStr += f"\n\tActivity: \n\t\t{self.inAct},\n\t\t{self.outAct}"
        return layStr

class FFFB(Net):
    '''A network with feed forward and feedback meshes between each
        layer. Based on ideas presented in [2]
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for index, layer in enumerate(self.layers[1:-1], 1): 
            #skip input and output layers, add feedback matrices
            nextLayer = self.layers[index+1]
            layer.addMesh(fbMesh(nextLayer.meshes[0], nextLayer))


if __name__ == "__main__":
    from .learningRules import GeneRec
    
    from sklearn import datasets
    import matplotlib.pyplot as plt

    net = FFFB([
        Layer(4, isInput=True),
        Layer(4, learningRule=GeneRec),
        Layer(4, learningRule=GeneRec)
    ], Mesh)

    iris = datasets.load_iris()
    inputs = iris.data
    maxMagnitude = np.max(np.sqrt(np.sum(np.square(inputs), axis=1)))
    inputs = inputs/maxMagnitude # bound on (0,1]
    targets = np.zeros((len(inputs),4))
    targets[np.arange(len(inputs)), iris.target] = 1
    #shuffle both arrays in the same manner
    shuffle = np.random.permutation(len(inputs))
    inputs, targets = inputs[shuffle], targets[shuffle]

    result = net.Learn(inputs, targets, numEpochs=500)
    plt.plot(result)
    plt.show()