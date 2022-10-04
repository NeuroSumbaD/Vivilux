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
from activations import Sigmoid
from metrics import RMSE
from learningRules import CHL

# library constants
DELTA_TIME = 0.1

class Net:
    '''Base class for neural networks with Hebbian-like learning
    '''
    def __init__(self, layers: iter, metric = RMSE) -> None:
        '''Instanstiates an ordered list of layers that will be
            applied sequentially during inference.
        '''
        self.layers = layers
        self.metric = metric

    def Predict(self, data):
        '''Inference method called 'prediction' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        # outputs = []
        self.layers[0].Clamp(data)

        for layer in self.layers[1:-1]:
            layer.Predict()

        output = self.layers[-1].Predict()
        
        return output

    def Observe(self, inData, outData):
        '''Training method called 'observe' in accordance with a predictive
            error-driven learning scheme of neural network computation.
        '''
        self.layers[0].Clamp(inData)
        for layer in self.layers[1:-1]:
            layer.Observe()

        self.layers[-1].Clamp(outData)

        return None # observations know the outcome

    def Infer(self, inData, numTimeSteps=25):
        for inDatum in inData:
            for time in range(numTimeSteps):
                self.Predict(inDatum)

    
    def Learn(self, inData, outData, numTimeSteps=50, numEpochs=50):
        results = np.zeros(numEpochs)
        epochResult = np.zeros(len(inData))
        for epoch in range(numEpochs):
            # iterate through data and time
            index=0
            for inDatum, outDatum in zip(inData, outData):
                for time in range(numTimeSteps):
                    lastResult = self.Predict(inDatum)
                    self.Observe(inDatum, outDatum)
                epochResult[index] = lastResult
                index += 1
            # update meshes
            for layer in self.layers:
                layer.Learn()
            # evaluate metric
            results[epoch] = self.metric(epochResult, outData)
        
        return results

class Mesh:
    '''Base class for meshes of synaptic elements.
    '''
    def __init__(self, size: int, layer, learningRate=0.1):
        self.size = size
        self.matrix = np.eye(size),
        self.inLayer = layer
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

    def Observe(self, data):
        data = self.inLayer.obsAct
        return self.apply(data)

    def Update(self, delta):
        self.matrix += self.rate*delta

class fbMesh(Mesh):
    '''A class for feedback meshes based on the transpose of another mesh.
    '''
    def __init__(self, mesh: Mesh) -> None:
        super.__init__(mesh.size)
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
        self.preAct = np.zeros(length)
        self.obsAct = np.zeros(length)
        self.act = activation
        self.rule = learningRule
        self.meshes = [] #empty initial mesh list

    def addMesh(self, mesh):
        self.meshes.append(mesh)

    def Predict(self):
        linAct = np.zeros(len(self))
        for mesh in self.meshes:
            linAct += mesh.Predict()
        self.preAct += DELTA_TIME*(self.act(linAct)-self.preAct)
        return self.preAct

    def Observe(self):
        linAct = np.zeros(len(self))
        for mesh in self.meshes:
            linAct += mesh.Observe()

        self.obsAct += DELTA_TIME*(self.act(linAct)-self.obsAct)
        return self.preAct

    def Clamp(self, data):
        self.obsAct = data

    def Learn(self):
        inLayer = self.meshes[0].inLayer # assume first mesh as input
        delta = self.rule(inLayer, self)
        self.meshes[0].Update(delta)

    def __len__(self):
        return len(self.preAct)

class FFFB(Net):
    '''A network with feed forward and feedback meshes between each
        layer. Based on ideas presented in [2]
    '''
    def __init__(self, layers: iter) -> None:
        super().__init__(layers)
        for index, layer in enumerate(self.layers[:-1]):
            layer.addMesh(fbMesh(self.layers[index+1].meshes[0]))


if __name__ == "__main__":
    from learningRules import GeneRec

    net = FFFB([4, 4])