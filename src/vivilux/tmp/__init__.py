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
    def __init__(self, layers: iter, meshType, metric = RMSE, learningRate = 0.1):
        '''Instanstiates an ordered list of layers that will be
            applied sequentially during inference.
        '''
        # TODO: allow different mesh types between layers
        self.layers = layers
        self.metric = metric

        for index, layer in enumerate(self.layers[1:]):
            size = len(layer)
            layer.addMesh(meshType(size, self.layers[index-1], learningRate))

    def Predict(self, data):
        '''Inference method called 'prediction' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        # outputs = []
        self.layers[0].Clamp(data)
        counter_layer = 0
        # print("here0")
        for layer in self.layers[0:-1]:
            # print("here1")
            counter_layer += 1
            # print(f"{counter_layer=}")
            layer.Predict()

        output = self.layers[-1].Predict()
        
        return output

    def Observe(self, inData, outData):
        '''Training method called 'observe' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        self.layers[0].Clamp(inData)
        counter_layer = 0
        for layer in self.layers[0:-1]:
            counter_layer += 1
            # print(f"{counter_layer=}")
            layer.Observe()

        self.layers[-1].Clamp(outData)

        return None # observations know the outcome

    def Infer(self, inData, numTimeSteps=25):
        for inDatum in inData:
            for time in range(numTimeSteps):
                self.Predict(inDatum)

    
    def Learn(self, inData, outData, numTimeSteps=50, numEpochs=50, verbose = False):
        results = np.zeros(numEpochs)
        epochResults = np.zeros((len(outData), len(self.layers[-1])))
        for iinx, layer in enumerate(list(self.layers)):
                print(">> pre-training mesh of layer ", iinx, " is ", layer.meshes[0])
        for epoch in range(numEpochs):
            # iterate through data and time
            index=0
            for inDatum, outDatum in zip(inData, outData): #>1
                #>2
                for time in range(numTimeSteps):#>3
                    lastResult = self.Predict(inDatum)
                    #>4
                for time in range(numTimeSteps):#>3
                    #>5
                    self.Observe(inDatum, outDatum)
                    #>6
                if index == 0:
                    print("inference of the first data sample: ", lastResult)
                epochResults[index] = lastResult
                index += 1
                if epoch != 0:
                    for layer in self.layers:
                        layer.Learn()
            # update meshes
            # if epoch != 0:
            #     for layer in self.layers:
            #         layer.Learn()
            # for iinx, layer in enumerate(list(self.layers)):
            #     print(">>for epoch=",epoch,"mesh of layer ", iinx, " is ", layer.meshes[0])
            # evaluate metric
            results[epoch] = self.metric(epochResults, outData)
            if verbose: print(self)
        
        return results

    def getWeights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.meshes[0].get())
        return weights

    def setLearningRule(self, rule):
        '''Sets the learning rule for all forward meshes to 'rule'.
        '''
        for layer in self.layers:
            layer.rule = rule

    def __str__(self) -> str:
        strs = []
        for layer in self.layers:
            strs.append(str(layer))

        return str(strs)

class Mesh:
    '''Base class for meshes of synaptic elements.
    '''
    def __init__(self, size: int, inLayer, learningRate=0.5):
        self.size = size if size > len(inLayer) else len(inLayer)
        self.matrix = np.eye(self.size)
        self.inLayer = inLayer
        self.rate = learningRate

    def set(self, matrix):
        self.matrix = matrix

    def get(self):
        return self.matrix

    def apply(self, data):
        try:
            return self.matrix @ data
        except ValueError as ve:
            print(f"Attempted to apply {data} (shape: {data.shape}) to mesh "
                  f"of dimension: {self.matrix}")

    def Predict(self):
        data = self.inLayer.preAct
        return self.apply(data)

    def Observe(self, data=0):
        data = self.inLayer.obsAct
        return self.apply(data)

    def Update(self, delta):
        self.matrix += self.rate*delta

    def __len__(self):
        return self.size

    def __str__(self):
        return f"\n\t\tMesh ({self.size}) = {self.get()}"

class fbMesh(Mesh):
    '''A class for feedback meshes based on the transpose of another mesh.
    '''
    def __init__(self, mesh: Mesh, inLayer) -> None:
        super().__init__(mesh.size, inLayer)
        self.mesh = mesh

    def set(self):
        raise Exception("Feedback mesh has no 'set' method.")

    def get(self):
        return self.mesh.matrix.T

    def apply(self, data):
        matrix = self.mesh.matrix.T
        try:
            return matrix @ data
        except ValueError as ve:
            print(f"Attempted to apply {data} (shape: {data.shape}) to mesh of dimension: {matrix}")

    def Update(self, delta):
        return None

class Layer:
    '''Base class for a layer that includes input matrices and activation
        function pairings. Each layer retains a seperate state for predict
        and observe phases, along with a list of input meshes applied to
        incoming data.
    '''
    def __init__(self, length, activation=Sigmoid, learningRule=CHL):
        self.preLin = np.zeros(length)
        self.preAct = np.zeros(length)
        
        self.obsLin = np.zeros(length)
        self.obsAct = np.zeros(length)
        self.act = activation
        self.rule = learningRule
        self.meshes = [] #empty initial mesh list

    def addMesh(self, mesh):
        self.meshes.append(mesh)

    def Predict(self):
        self.preLin -= DELTA_TIME*self.preLin
        counterr =0
        for mesh in self.meshes:
            counterr +=1
            # print(f"{counterr=}")
            self.preLin += DELTA_TIME * mesh.Predict()[:len(self)]**2
        self.preAct = self.act(self.preLin)
        return self.preAct

    def Observe(self):
        self.obsLin -= DELTA_TIME * self.obsLin
        counterr = 0
        for mesh in self.meshes:
            counterr +=1
            # print(f"{counterr=}")
            self.obsLin += DELTA_TIME * mesh.Observe()[:len(self)]**2
        self.obsAct = self.act(self.obsLin)
        return self.preAct

    def Clamp(self, data):
        self.obsAct = data[:len(self)]

    def Learn(self):
        inLayer = self.meshes[0].inLayer # assume first mesh as input
        delta = self.rule(inLayer, self)
        self.meshes[0].Update(delta)

    def __len__(self):
        return len(self.preAct)

    def __str__(self) -> str:
        str = f"Layer ({len(self)}): \n\tActivation = {self.act}\n\tLearning"
        str += f"Rule = {self.rule}"
        str += f"\n\tMeshes: {self.meshes}"
        return str

class FFFB(Net):
    '''A network with feed forward and feedback meshes between each
        layer. Based on ideas presented in [2]
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for index, layer in enumerate(self.layers[:-1]):
            nextLayer = self.layers[index+1]
            layer.addMesh(fbMesh(nextLayer.meshes[0], nextLayer))


if __name__ == "__main__":
    from .learningRules import GeneRec
    
    from sklearn import datasets
    import matplotlib.pyplot as plt

    net = FFFB([
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

    result = net.Learn(inputs, targets, numEpochs=5000)
    plt.plot(result)
    plt.show()