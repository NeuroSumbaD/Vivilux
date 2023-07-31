'''
A library for Hebbian-like learning implementations on MZI meshes based on the
work of O'Reilly et al. [1] in computational neuroscience (see https://github.com/emer/leabra).

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
from .optimizers import ByPass
from .monitors import Monitor

# library default constants
DELTA_TIME = 0.1

class Net:
    '''Base class for neural networks with Hebbian-like learning
    '''
    count = 0
    def __init__(self, layers: list[Layer], meshType: Mesh,
                 metric = RMSE, learningRate = 0.1, name = None,
                 optimizer = ByPass(),
                 meshArgs = {},
                 numTimeSteps = 50,
                 monitoring = False,
                 **kwargs):
        '''Instanstiates an ordered list of layers that will be
            applied sequentially during inference.
        '''
        self.DELTA_TIME = DELTA_TIME
        self.numTimeSteps = numTimeSteps
        self.monitoring = monitoring

        self.name =  f"NET_{Layer.count}" if name == None else name
        Net.count += 1

        # TODO: allow different mesh types between layers
        self.layers = layers
        self.metrics = [metric] if not isinstance(metric, list) else metric

        for index, layer in enumerate(self.layers[1:], 1):
            size = len(layer)
            layer.addMesh(meshType(size, self.layers[index-1],
                                   learningRate,
                                   **meshArgs))
            layer.optimizer = optimizer
            layer.monitor = Monitor(name = self.name + ": " + layer.name,
                                    labels = ["time step", "activity"],
                                    limits=[numTimeSteps, 1],
                                    numLines=len(layer))

    def Predict(self, data):
        '''Inference method called 'prediction' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        #Clamp input layer, set minus phase history
        self.layers[0].Clamp(data)
        self.layers[0].phaseHist["minus"][:] = self.layers[0].getActivity()

        for layer in self.layers[1:-1]:
            layer.Predict()
            
        output = self.layers[-1].Predict()
        
        
        return output

    def Observe(self, inData, outData):
        '''Training method called 'observe' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        #Clamp input layer, set minus phase history
        self.layers[0].Clamp(inData)
        self.layers[0].phaseHist["plus"][:] = self.layers[0].getActivity()
        #Clamp output layer, set minus phase history
        self.layers[-1].Clamp(outData)
        self.layers[-1].phaseHist["plus"][:] = self.layers[-1].getActivity()
        
        for layer in self.layers[1:-1]:
            layer.Observe()
        # self.layers[-1].ClampObs(outData)

        return None # observations know the outcome

    def Infer(self, inData, numTimeSteps=50):
        outputData = np.zeros((len(inData), len(self.layers[-1])))
        index = 0
        for inDatum in inData:
            for time in range(numTimeSteps):
                result = self.Predict(inDatum)
            outputData[index][:] = result
            index += 1
        return outputData

    
    def Learn(self, inData: np.ndarray, outData: np.ndarray,
              numTimeSteps = None, numEpochs=50,
              verbose = False, reset = False):
        '''Control loop for learning based on GeneRec-like algorithms.
                inData      : input data
                outData     : 
                verbose     : if True, prints net each iteration
                reset       : if True, resets activity between each input sample
        '''
        # allow update of numTimeSteps
        if numTimeSteps is not None:
            self.numTimeSteps = numTimeSteps

        results = np.zeros(numEpochs+1)
        index = 0
        numSamples = len(inData)

        print("Progress:")
        print(f"Epoch: 0, sample: ({index}/{numSamples}), metric[{self.metrics[0].__name__}] = {results[0]:0.2f}  ", end="\r")
        results[0] = self.Evaluate(inData, outData, self.numTimeSteps)
        print(f"Epoch: 0, sample: ({index}/{numSamples}), metric[{self.metrics[0].__name__}] = {results[0]:0.2f}  ", end="\r")

        epochResults = np.zeros((len(outData), len(self.layers[-1])))
        
        for epoch in range(numEpochs):
            # iterate through data and time
            index=0
            for inDatum, outDatum in zip(inData, outData):
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
                print(f"Epoch: ({epoch}/{numEpochs}), sample: ({index}/{numSamples}), metric[{self.metrics[0].__name__}] = {results[epoch]:0.4f}  ", end="\r")
            # evaluate metric
            #TODO: record multiple metrics
            results[epoch+1] = self.metrics[0](epochResults, outData)
            print(f"Epoch: ({epoch}/{numEpochs}), sample: ({index}/{numSamples}), metric[{self.metrics[0].__name__}] = {results[epoch+1]:0.4f}  ", end="\r")
            if verbose: print(self)
        print("\n")
        return results
    
    def Evaluate(self, inData, outData, numTimeSteps=25):
        results = self.Infer(inData, numTimeSteps)
        return self.metrics[0](results, outData)

    def getWeights(self, ffOnly):
        weights = []
        for layer in self.layers:
            for mesh in layer.meshes:
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

class Mesh:
    '''Base class for meshes of synaptic elements.
    '''
    count = 0
    def __init__(self, size: int, inLayer: Layer,
                 learningRate=0.5,
                 **kwargs):
        self.size = size if size > len(inLayer) else len(inLayer)
        self.matrix = np.eye(self.size)
        self.inLayer = inLayer
        self.rate = learningRate

        # flag to track when matrix updates (for nontrivial meshes like MZI)
        self.modified = False

        self.name = f"MESH_{Mesh.count}"
        Mesh.count += 1

    def set(self, matrix):
        self.modified = True
        self.matrix = matrix

    def get(self):
        return self.matrix
    
    def getInput(self):
        return self.inLayer.getActivity()

    def apply(self):
        data = self.getInput()
        # guarantee that data can be multiplied by the mesh
        data = np.pad(data[:self.size], (0, self.size - len(data)))
        return self.applyTo(data)
            
    def applyTo(self, data):
        try:
            return self.get() @ data
        except ValueError as ve:
            print(f"Attempted to apply {data} (shape: {data.shape}) to mesh "
                  f"of dimension: {self.matrix}")

    def Update(self, delta: np.ndarray):
        m, n = delta.shape
        self.modified = True
        self.matrix[:m, :n] += self.rate*delta

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
    
class InhibMesh(Mesh):
    '''A class for inhibitory feedback mashes based on fffb mechanism.
        Calculates inhibitory input to a layer based on a mixture of its
        existing activation and current input.
    '''
    FF = 1
    FB = 1
    FBTau = 1/1.4
    FF0 = 0.1

    def __init__(self, ffmesh: Mesh, inLayer: Layer) -> None:
        self.name = "FFFB_" + ffmesh.name
        self.ffmesh = ffmesh
        self.size = len(inLayer)
        self.inLayer = inLayer
        self.fb = 0
        self.inhib = np.zeros(self.size)

    def apply(self):
        # guarantee that data can be multiplied by the mesh
        ffAct = self.ffmesh.apply()[:len(self)]
        ffAct = np.pad(ffAct, (0, self.size - len(ffAct)))
        ffAct = np.maximum(ffAct-InhibMesh.FF0,0)

        self.fb += InhibMesh.FBTau * (np.mean(self.inLayer.outAct) - self.fb)

        self.inhib[:] = InhibMesh.FF * ffAct + InhibMesh.FB * self.fb
        return -self.inhib

    def set(self):
        raise Exception("InhibMesh has no 'set' method.")

    def get(self):
        return self.apply()
    
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
        self.inAct = np.zeros(length) # linearly integrated dendritic inputs (internal Activation)
        self.outAct = activation(self.inAct) #initialize outgoing Activation
        self.modified = False 
        self.phaseHist = {"minus": np.zeros(length),
                          "plus": np.zeros(length)
                          }
        
        self.act = activation
        self.rule = learningRule
        self.meshes: list[Mesh] = [] #empty initial mesh list

        self.optimizer = ByPass()

        self.isInput = isInput
        self.freeze = False
        self.name =  f"LAYER_{Layer.count}" if name == None else name
        if isInput: self.name = "INPUT_" + self.name

        self.monitor = None

        Layer.count += 1

    def getActivity(self):
        if self.modified == True:
            self.outAct[:] = self.act(self.inAct)
            self.modified = False
        return self.outAct

    def printActivity(self):
        return [self.inAct, self.outAct]
    
    def resetActivity(self):
        '''Resets all activation traces to zero vectors.'''
        length = len(self)
        self.inAct = np.zeros(length)
        self.outAct = np.zeros(length)

    def Integrate(self):
        for mesh in self.meshes:
            self += DELTA_TIME * mesh.apply()[:len(self)]#**2

    def Predict(self):
        self -= DELTA_TIME*self.inAct
        self.Integrate()
        activity = self.getActivity()
        self.phaseHist["minus"][:] = activity
        self.monitor.update(activity)
        return activity.copy()

    def Observe(self):
        self -= DELTA_TIME * self.inAct
        self.Integrate()
        activity = self.getActivity()
        self.phaseHist["plus"][:] = activity
        return activity.copy()

    def Clamp(self, data):
        self.inAct[:] = data[:len(self)]
        self.outAct[:] = data[:len(self)]

    def Learn(self):
        if self.isInput or self.freeze: return
        # TODO: Allow multiple meshes to learn, skip fb meshes
        inLayer = self.meshes[0].inLayer # assume first mesh as input
        delta = self.rule(inLayer, self)
        optDelta = self.optimizer(delta)
        self.meshes[0].Update(optDelta)

        
    def Freeze(self):
        self.freeze = True

    def Unfreeze(self):
        self.freeze = False
    
    def addMesh(self, mesh):
        self.meshes.append(mesh)

    def __add__(self, other):
        self.modified = True
        return self.inAct + other
    
    def __radd__(self, other):
        self.modified = True
        return self.inAct + other
    
    def __iadd__(self, other):
        self.modified = True
        self.inAct += other
        return self
    
    def __sub__(self, other):
        self.modified = True
        return self.inAct - other
    
    def __rsub__(self, other):
        self.modified = True
        return self.inAct - other
    
    def __isub__(self, other):
        self.modified = True
        self.inAct -= other
        return self
    
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
    def __init__(self, *args, FeedbackMesh = fbMesh, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for index, layer in enumerate(self.layers[1:-1], 1): 
            #skip input and output layers, add feedback matrices
            nextLayer = self.layers[index+1]
            layer.addMesh(FeedbackMesh(nextLayer.meshes[0], nextLayer))
            inhibitoryMesh = InhibMesh(layer.meshes[0], layer)
            layer.addMesh(inhibitoryMesh)


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