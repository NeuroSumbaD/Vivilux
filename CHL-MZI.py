# -*- coding: utf-8 -*-
"""
Initially Created on Thu May 31 2022

@author: Luis El Srouji
"""

from itertools import product
import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import svd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
np.random.seed(seed=420)


def Sigmoid(input):
    return 1/(1 + np.exp(-10*(input-0.5)))

def Detect(input):
    '''DC power detected (no cross terms)
    '''
    return np.square(np.sum(input, axis=-1))

def Diagonalize(vector):
    diag = np.eye(len(vector))
    for i in range(len(vector)):
        diag[i,i] = vector[i]
    return diag

def DeltaTheta(deltaLayers, allStages):
    deltaThetas = [ [np.zeros(len(deltaWeights)) for deltaWeights in deltaStages
                    ] for deltaStages in deltaLayers
                  ]
    for layer, (deltaStages, stages) in enumerate(zip(deltaLayers, allStages)):
        for stage in stages:
            for thetaIndex, theta in enumerate(stages[stage]["vars"]):
                delta = deltaStages[stage][thetaIndex]
                deltaThetas[layer][stage][thetaIndex] = (delta[0,0]/np.cos(theta)
                                                                 - delta[0,1]/np.sin(theta)
                                                                 - delta[1,0]/np.sin(theta)
                                                                 - delta[1,1]/np.cos(theta)
                                                                  )
    return deltaThetas

def BoundTheta(thetas):
    thetas[thetas > (2*np.pi)] -= 2*np.pi
    thetas[thetas < 0] += 2*np.pi  

class MZImesh:
    def __init__(self, dimension):
        self.dimension = dimension
        self.numUnits = int(dimension*(dimension-1)/2)
        self.phaseShifters = np.random.rand(self.numUnits)*2*np.pi

    def getUnits(self, units=None):
        if units is not None:
            numUnits = len(units)
            return np.array([[[np.sin(theta),np.cos(theta)],
                              [np.cos(theta),-np.sin(theta)]]
                              for theta in units.flatten()]).reshape(numUnits,-1,2,2)
        else:
            return np.array([[[np.sin(theta),np.cos(theta)],
                              [np.cos(theta),-np.sin(theta)]]
                              for theta in self.phaseShifters])
        
    def getStages(self, startIndex = 0):
        '''Returns a dictionary with integer keys for each stage
            and correctly dimensioned matrices for each stage.
        '''
        stages = {}
        index = startIndex #index in phase shifters
        mziUnits = self.getUnits()
        units = self.dimension//2
        for stage in range(self.dimension):
            parity = stage % 2 #even or odd stage
            stages[stage] = {}
            stages[stage]["index"] = index
            stages[stage]["units"] = []
            numUnits = (self.dimension-parity)//2
            stages[stage]["vars"] = self.phaseShifters[index:index+numUnits]
            for unit in range(units):
                currUnit = np.eye(self.dimension)
                wg = 2*unit+parity #waveguide number
                if wg < self.dimension - 1: #bound by number of waveguides
                    # stages[stage]["vars"].append(self.phaseShifters[index])
                    currUnit[wg:wg+2,wg:wg+2] = mziUnits[index]
                    stages[stage]["units"].append(currUnit)
                    index += 1
            # stages[stage]["vars"] = np.array(stages[stage]["vars"])
        return stages
    
    def getMatrix(self):
        '''Returns full mesh matrix.
        '''
        fullMatrix = np.eye(self.dimension)
        stages = self.getStages()
        for stage in stages:
            for unit in stages[stage]["units"]:
                fullMatrix = unit @ fullMatrix
        return fullMatrix

    def getFeedback(self):
        '''Returns transpose of mesh matrix.
        '''
        return self.getMatrix().T

    def split(self, stage):
        '''Returns weight matrices with a split after
           a certain stage.
        '''
        before = np.eye(self.dimension)
        current = np.eye(self.dimension)
        after = np.eye(self.dimension)
        stages = self.getStages()
        #matrix up to just before indicated stage
        for stageIndex in range(stage-1):
            for unit in stages[stageIndex]["units"]:
                before = unit @ before
        #matrix just for current stage
        for unit in stages[stage]["units"]:
            current = unit @ current
        #matrix after indicated stage
        for stageIndex in range(stage+1, self.dimension):
            for unit in stages[stageIndex]["units"]:
                after = unit @ after
        return before, current, after


class OpticalNetwork:
    def __init__(self, dimensions, MeshType = MZImesh):
        '''Initialize a photonic neural network of MZI meshes.
            "Dimensions" is a list with the dimensions of each layer.
        '''
        self.numLayers = len(dimensions)
        self.dimensions = dimensions
        self.layers = []
        
        # Between two layers, the MZI mesh with have the dimension
        ### of the larger one.
        for index in range(self.numLayers):
            dimension = dimensions[index]
            self.layers.append({"index": index,
                                "dimension": dimension,
                                "rateCode": Sigmoid,
                                "mAct": np.zeros(dimension),
                                "pAct": np.zeros(dimension),
                                "mesh": MZImesh(dimension)
                                })
            

    def Predict(self, sample, numTimeSteps = 50, deltaTime = 0.1):
        '''Predict expected output from input sample.
            Activity corresponds to minus phase of CHL algorithm.
        '''
        for timeStep in range(numTimeSteps):
            for index, layer in enumerate(self.layers):
                # main inference update
                input = sample if index == 0 else self.layers[index-1]["mAct"]
                input = Diagonalize(input)
                detectorIn = layer["mesh"].getMatrix() @ input
                if index != self.numLayers - 1:
                    nextLayer = self.layers[index+1]
                    detectorIn += nextLayer["mesh"].getFeedback() @ Diagonalize(nextLayer["mAct"])
                detectorOut = Detect(detectorIn)
                layer["mAct"] += deltaTime*(layer["rateCode"](detectorOut)-layer["mAct"])

    def Observe(self, sample, target, numTimeSteps = 50, deltaTime = 0.1):
        '''Clamp to expected output associated with input sample.
            Activity corresponds to plus phase of CHL algorithm.
        '''
        for timeStep in range(numTimeSteps):
            for index, layer in enumerate(self.layers):
                if index == self.numLayers - 1:
                    layer["pAct"] = target
                    continue #escape loop, save time
                # main inference update
                input = sample if index == 0 else self.layers[index-1]["pAct"]
                input = Diagonalize(input)
                detectorIn = layer["mesh"].getMatrix() @ input
                nextLayer = self.layers[index+1]
                detectorIn += nextLayer["mesh"].getFeedback() @ Diagonalize(nextLayer["pAct"])
                detectorOut = Detect(detectorIn)
                layer["pAct"] += deltaTime*(layer["rateCode"](detectorOut)-layer["pAct"])

    def Train(self, samples, targets, numEpochs = 100, numTimesteps=50, deltaTime = 0.1, learningRate = 0.1):
        '''Updates the MZI meshes to drive the mesh towards its the
            target activities.
        '''
        epochErrors = np.zeros(numEpochs)
        for epoch in range(numEpochs):
            errors = np.zeros((numSamples, self.dimensions[-1]))
            for sampleIndex, (sample, target) in enumerate(zip(samples, targets)):
                allStages = [layer["mesh"].getStages() for layer in self.layers]
                deltaThetas = DeltaTheta(self.CHL(sample, target, numTimesteps, deltaTime, learningRate), allStages)
                if epoch != 0 : # Don't train on first epoch
                    for layerStages, deltaLayer in zip(allStages, deltaThetas):
                        for stage, deltaStage in zip(layerStages, deltaLayer):
                            layerStages[stage]["vars"] += deltaStage
                            layerStages[stage]["vars"] = BoundTheta(layerStages[stage]["vars"])
                lastLayer = self.layers[-1]
                errors[sampleIndex] = np.sum(np.square(lastLayer["mAct"]-target))
            epochErrors[epoch] = np.sqrt(np.mean(errors))
        return epochErrors

    def Evaluate(self, samples, targets, numEpochs = 100, numTimesteps=50, deltaTime = 0.1):
        '''Iterate through the dataset but do not update the meshes'''
        epochErrors = np.zeros(numEpochs)
        for epoch in range(numEpochs):
            errors = np.zeros((numSamples, self.dimensions[-1]))
            for sampleIndex, (sample, target) in enumerate(zip(samples, targets)):
                self.Predict(sample, numTimesteps, deltaTime)
                lastLayer = self.layers[-1]
                errors[sampleIndex] = np.sum(np.square(lastLayer["mAct"]-target))
            epochErrors[epoch] = np.sqrt(np.mean(errors))
        return epochErrors

    def CHL(self, sample, target, numTimesteps=50, deltaTime=0.1, learningRate = 0.1):
        deltaLayers = []
        self.Predict(sample, numTimesteps, deltaTime)
        self.Observe(sample, target, numTimesteps, deltaTime)
        for layerIndex, layer in enumerate(self.layers):
            deltaStages = []
            stages = layer["mesh"].getStages()
            for stage in stages:
                before, current, after = layer["mesh"].split(stage)
                # Calculate minus phase activity
                mInput = sample if layer["index"] == 0 else self.layers[layerIndex-1]["mAct"]
                mInput = Diagonalize(mInput)
                mIn = Detect(before @ mInput)
                mIn = self.Even(mIn, stage)
                mOut = Detect(current @ before @ mInput)
                mOut = self.Even(mOut, stage)
                # Calculate plus phase activity
                pInput = sample if layer["index"] == 0 else self.layers[layerIndex-1]["pAct"]
                pInput = Diagonalize(pInput)
                pIn = Detect(before @ pInput)
                pIn = self.Even(pIn, stage)
                pOut = Detect(current @ before @ pInput)
                pOut = self.Even(pOut, stage)
                # Calculate delta according to CHL
                delta = pOut.reshape(-1,2,1) @ pIn.reshape(-1,1,2)
                delta -= mOut.reshape(-1,2,1) @ mIn.reshape(-1,1,2)
                delta *= learningRate
                deltaStages.append(delta)
            deltaLayers.append(deltaStages)
        return deltaLayers

    def Even(self, vector, stage):
        '''Returns the correct number of even paired waveguides corresponding
            to MZI inputs at a given stage.
        '''
        parity = stage % 2
        vector = vector[parity:]
        vector = vector[:len(vector)//2*2]
        return vector

    def Evaluate(self, samples, targets):
        pass


def trainingLoopCHL(W1, W2, inputs, targets, numEpochs=100,
                    numTimeSteps=100, phaseStep = 50, learningRate = 0.1,
                    deltaTime = 0.1, plot=True):
    '''CHL:
        Training using Contrastive Hebbian Learning rule
    '''
    numSamples = len(inputs)
    #Allocate error traces
    fullErrorTrace = np.zeros(numTimeSteps*numSamples*numEpochs)
    errorTrace = np.zeros(numEpochs)
    
    #allocate space for variables during learning
    print("Allocating space for loop variables...")
    matrixDimension = len(W1)
    linInp = np.zeros(matrixDimension) #linear input layer
    actInp = np.zeros(matrixDimension)
    linOut = np.zeros(matrixDimension)
    actOut = np.zeros(matrixDimension)
    minusPhaseIn = np.zeros(matrixDimension)
    minusPhaseOut = np.zeros(matrixDimension)
    minusPhaseIn = np.zeros(matrixDimension)
    minusPhaseOut = np.zeros(matrixDimension)
    weightIn = W1.copy()
    weightOut = W2.copy()
    print("Beginning training...")

    for epoch in range(numEpochs):
        epochErrors = np.zeros(numSamples)
        for sample in range(numSamples):
            currentInput = inputs[sample]
            targetOutput = targets[sample]

            for timeStep in range(numTimeSteps):
                #update activation values
                linInp += deltaTime*(np.abs(weightIn @ currentInput)**2
                                   + np.abs(weightOut.T @ actOut)**2
                                   - linInp)
                actInp = Sigmoid(linInp)
                if timeStep <= phaseStep:
                    linOut += deltaTime*np.abs(weightOut @ actInp)**2-linOut
                    actOut = Sigmoid(linOut)
                    if timeStep == phaseStep:
                        minusPhaseIn = actInp
                        minusPhaseOut = actOut
                        epochErrors[sample] = np.sum((targetOutput - actOut)**2)
                else:
                    actOut = targetOutput
                
                #Record traces
                traceIndex = epoch*(numSamples*numTimeSteps)+sample*numTimeSteps + timeStep
                # inputTrace[traceIndex] = actInp
                # outputTrace[traceIndex] = actOut
                # weightInTrace[traceIndex] = weightIn.flatten()
                # weightOutTrace[traceIndex] = np.abs(weightOut).flatten()
                fullErrorTrace[traceIndex] = np.sqrt(np.sum((targetOutput - actOut)**2))
            
            plusPhaseIn = actInp
            plusPhaseOut = actOut
            if epoch != 0: # don't train on first epoch to establish RMSE
                #Contrastive Hebbian Learning rule
                ####(equivalent to GenRec with symmetry and midpoint approx)
                ######## (generally converges faster)
                deltaWeightIn = (plusPhaseIn[:,np.newaxis] @ plusPhaseOut[np.newaxis,:] -
                                minusPhaseIn[:,np.newaxis] @ minusPhaseOut[np.newaxis,:])
                # weightIn += learningRate * deltaWeightIn # FIXME FREEZE FIRST LAYER
                deltaWeightOut = (plusPhaseOut - minusPhaseOut)[:,np.newaxis] @ minusPhaseIn[np.newaxis,:] 

                weightOut += learningRate * deltaWeightOut
        
        #Store RMSE for the given epoch
        errorTrace[epoch] = np.sqrt(np.mean(epochErrors))

    print("Done")


    # print("final input weight matrix:\n", weightIn)
    # print("final output weight matrix:\n", weightOut)
    print(f"initial RMSE: {errorTrace[0]}, final RMSE: {errorTrace[-1]}")
    print(f"Training occured?: {errorTrace[0] > errorTrace[-1]}")

    if plot:
        plt.figure()
        nonZeroIndices = fullErrorTrace != 0
        plt.plot(np.arange(len(fullErrorTrace))[nonZeroIndices]/numSamples/numTimeSteps,
                 fullErrorTrace[nonZeroIndices], "r")
        plt.plot(range(len(errorTrace)), errorTrace, "b--")
        plt.title("Error")
        plt.ylabel("RMSE")
        plt.xlabel("Epoch")
        plt.show()

    return errorTrace


def randomWeights(W1, W2, inputs, targets, numEpochs=100,
                    numTimeSteps=100, phaseStep = 50, learningRate = 0.1,
                    deltaTime = 0.1, plot=True):
    '''CHL:
        Training using Contrastive Hebbian Learning rule
    '''
    numSamples = len(inputs)
    #Allocate error traces
    fullErrorTrace = np.zeros(numTimeSteps*numSamples*numEpochs)
    errorTrace = np.zeros(numEpochs)
    
    #allocate space for variables during learning
    print("Allocating space for loop variables...")
    matrixDimension = len(W1)
    linInp = np.zeros(matrixDimension) #linear input layer
    actInp = np.zeros(matrixDimension)
    linOut = np.zeros(matrixDimension)
    actOut = np.zeros(matrixDimension)
    minusPhaseIn = np.zeros(matrixDimension)
    minusPhaseOut = np.zeros(matrixDimension)
    minusPhaseIn = np.zeros(matrixDimension)
    minusPhaseOut = np.zeros(matrixDimension)
    weightIn = W1.copy()
    weightOut = W2.copy()
    print("Beginning training...")

    for epoch in range(numEpochs):
        epochErrors = np.zeros(numSamples)
        for sample in range(numSamples):
            currentInput = inputs[sample]
            targetOutput = targets[sample]

            for timeStep in range(numTimeSteps):
                #update activation values
                linInp += deltaTime*(np.abs(weightIn @ currentInput)**2
                                   + np.abs(weightOut.T @ actOut)**2
                                   - linInp)
                actInp = Sigmoid(linInp)
                if timeStep <= phaseStep:
                    linOut += deltaTime*np.abs(weightOut @ actInp)**2-linOut
                    actOut = Sigmoid(linOut)
                    if timeStep == phaseStep:
                        minusPhaseIn = actInp
                        minusPhaseOut = actOut
                        epochErrors[sample] = np.sum((targetOutput - actOut)**2)
                else:
                    actOut = targetOutput
                
                #Record traces
                traceIndex = epoch*(numSamples*numTimeSteps)+sample*numTimeSteps + timeStep
                # inputTrace[traceIndex] = actInp
                # outputTrace[traceIndex] = actOut
                # weightInTrace[traceIndex] = weightIn.flatten()
                # weightOutTrace[traceIndex] = np.abs(weightOut).flatten()
                fullErrorTrace[traceIndex] = np.sqrt(np.sum((targetOutput - actOut)**2))
            
            plusPhaseIn = actInp
            plusPhaseOut = actOut
            if epoch != 0: # don't train on first epoch to establish RMSE
                #Contrastive Hebbian Learning rule
                ####(equivalent to GenRec with symmetry and midpoint approx)
                ######## (generally converges faster)
                deltaWeightIn = (plusPhaseIn[:,np.newaxis] @ plusPhaseOut[np.newaxis,:] -
                                minusPhaseIn[:,np.newaxis] @ minusPhaseOut[np.newaxis,:])
                # weightIn += learningRate * deltaWeightIn # FIXME FREEZE FIRST LAYER
                # deltaWeightOut = (plusPhaseOut - minusPhaseOut)[:,np.newaxis] @ minusPhaseIn[np.newaxis,:] 
                deltaWeightOut = np.random.rand(4,4)
                weightOut += learningRate * deltaWeightOut
        
        #Store RMSE for the given epoch
        errorTrace[epoch] = np.sqrt(np.mean(epochErrors))

    print("Done")


    # print("final input weight matrix:\n", weightIn)
    # print("final output weight matrix:\n", weightOut)
    print(f"RANDOM initial RMSE: {errorTrace[0]}, final RMSE: {errorTrace[-1]}")
    print(f"Training occured?: {errorTrace[0] > errorTrace[-1]}")

    if plot:
        plt.figure()
        nonZeroIndices = fullErrorTrace != 0
        plt.plot(np.arange(len(fullErrorTrace))[nonZeroIndices]/numSamples/numTimeSteps,
                 fullErrorTrace[nonZeroIndices], "r")
        plt.plot(range(len(errorTrace)), errorTrace, "b--")
        plt.title("Error")
        plt.ylabel("RMSE")
        plt.xlabel("Epoch")
        plt.show()

    return errorTrace


def hyperparameterLoop(LearningRates, DeltaTimes, TrainingLoops,
                       numEpochs, numSamples, numTimeSteps, W1, W2,
                       Inputs, Targets):
    results = {}
    for training in TrainingLoops:
        for learningRate in LearningRates:
            for deltaTime in DeltaTimes:
                name = training.__doc__.split(":")[0]
                identifier = f"{name}_lr{learningRate}_dt{deltaTime}"
                print(f"Starting test: {identifier}")
                results[identifier] = training(numEpochs, numSamples, numTimeSteps, W1, W2,
                       Inputs, Targets, plot=False, animate=False)

    plt.figure()
    for key in results:
        plt.plot(range(len(results[key])), results[key], label=key)
    plt.title("Hyperparameter Results")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

    return results

if __name__ == "__main__":
    from sklearn import datasets

    iris = datasets.load_iris()
    inputs = iris.data
    maxMagnitude = np.max(np.sqrt(np.sum(np.square(inputs), axis=1)))
    inputs = inputs/maxMagnitude # bound on (0,1]
    targets = np.zeros((len(inputs),4))
    targets[np.arange(len(inputs)), iris.target] = 1
    #shuffle both arrays in the same manner
    shuffle = np.random.permutation(len(inputs))
    inputs, targets = inputs[shuffle], targets[shuffle]

    matrixDimension = 4
    numSamples = len(inputs)
    model  = OpticalNetwork([4,4,4])

    #define weight matrices
    #### there are two layers:
    ######## a hidden unit -> activation represents input to MZI mesh
    ######## an output unit -> activation is the output of MZI (after nonlinearity)
    weightIn = model.layers[0]["mesh"].getMatrix()
    weightOut = model.layers[1]["mesh"].getMatrix()

    #define input and output data (must be normalized and positive-valued)
    # vecs = np.random.normal(size=(numSamples, matrixDimension))
    # mags = np.linalg.norm(vecs, axis=-1)
    # inputs = np.abs(vecs/mags[...,np.newaxis])
    # vecs = np.random.normal(size=(numSamples, matrixDimension))
    # mags = np.linalg.norm(vecs, axis=-1)
    # targets = np.abs(vecs/mags[...,np.newaxis])
    # del vecs, mags
    # # random classes
    # targets = np.zeros((numSamples,4))
    # for entry, rand in zip(targets, np.floor(np.random.rand(numSamples)*4).astype("int")):
    #     entry[rand] = 1 



    resultMZI = model.Train(inputs, targets, numEpochs=500, learningRate=0.015)
    print(f"MZI initial RMSE: {resultMZI[0]}, final RMSE: {resultMZI[-1]}")
    print(f"MZI Training occured?: {resultMZI[0] > resultMZI[-1]}")

    # print("initial input weight matrix:\n", weightIn)
    # print("initial output weight matrix:\n", weightOut)
    resultCHL = trainingLoopCHL(weightIn, weightOut,inputs, targets, numEpochs=500, learningRate=0.05, plot=False)

    # resultRand = randomWeights(weightIn, weightOut,inputs, targets, numEpochs=500, learningRate=0.05, plot=False)
                
    plt.figure()
    plt.plot(resultMZI, label="MZI")
    plt.plot(resultCHL, label="CHL")
    # plt.plot(resultRand, label="Rand")
    plt.title("Simulated MZI Mesh Training")
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.show()
