import copy

from vivilux import *
from vivilux.learningRules import CHL, CHL2, GeneRec

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)

numSamples = 40

def CHL_trans(inLayer, outLayer):
    return CHL(inLayer, outLayer).T

#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
targets = np.abs(vecs/mags[...,np.newaxis])
del vecs, mags

netGR = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01)

netCHL = copy.deepcopy(netGR)
netCHL.setLearningRule(CHL2)

netCHL_T = copy.deepcopy(netGR)
netCHL_T.setLearningRule(CHL_trans)

netMixed = copy.deepcopy(netCHL)
netMixed.setLearningRule(GeneRec, 1) #sets second layer learnRule

print("DATA BEFORE TRAINING NETMIXED")
for layer in netMixed.layers:
    print(layer)
    for mesh in layer.meshes:
        print(mesh) 

def trainingLoopCHL(W1, W2, inputs, targets, numEpochs=100, numSamples=40,
                    numTimeSteps=100, phaseStep = 50, learningRate = 0.1,
                    deltaTime = 0.1):
    '''CHL:
        Training using Contrastive Hebbian Learning rule
    '''
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
                print(f"Timestep: {timeStep}")
                #update activation values
                linInp += deltaTime*(np.abs(weightIn @ currentInput)**2
                                   + np.abs(weightOut.T @ actOut)**2
                                   - linInp)
                actInp = Sigmoid(linInp)
                if timeStep <= phaseStep:
                    linOut += deltaTime*(np.abs(weightOut @ actInp)**2-linOut)
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
                print(f"\n{linInp}, {actInp}, {linOut}, {actOut}\n\n")
            
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
                print(f"\ndeltaWeightIn: {deltaWeightIn}\n\ndeltaWeightOut: {deltaWeightOut}")
                weightOut += learningRate * deltaWeightOut
        
        #Store RMSE for the given epoch
        errorTrace[epoch] = np.sqrt(np.mean(epochErrors))

    print("Done")


    # print("final input weight matrix:\n", weightIn)
    # print("final output weight matrix:\n", weightOut)
    print(f"initial RMSE: {errorTrace[0]}, final RMSE: {errorTrace[-1]}")
    print(f"Training occured?: {errorTrace[0] > errorTrace[-1]}")

    return errorTrace

weights = netGR.getWeights(ffOnly=True)

oldResult = trainingLoopCHL(weights[0], weights[1],inputs, targets, numEpochs=200, learningRate=0.1)
# plt.plot(oldResult, label="Old GR")

# print(f"net: {str(netGR)}")
# print(f"Initial {netGR.metric}: ", netGR.Evaluate(inputs, targets))
# resultGR = netGR.Learn(inputs, targets, numEpochs=200)
# print(f"Final {netGR.metric}: ", resultGR[-1])
# plt.plot(resultGR, label="GeneRec")

# print(f"net: {str(netCHL)}")
# print(f"Initial {netCHL.metric}: ", netCHL.Evaluate(inputs, targets))
# resultCHL = netCHL.Learn(inputs, targets, numEpochs=200)
# print(f"Final {netCHL.metric}: ", resultCHL[-1])
# plt.plot(resultCHL, label="CHL")

# print(f"net: {str(netCHL_T)}")
# print(f"Initial {netCHL_T.metric}: ", netCHL_T.Evaluate(inputs, targets))
# resultCHL_T = netCHL_T.Learn(inputs, targets, numEpochs=200)
# print(f"Final {netCHL_T.metric}: ", resultCHL_T[-1])
# plt.plot(resultCHL_T, label="CHL_T")

print(f"net: {str(netMixed)}")
initial = netMixed.Evaluate(inputs, targets)
print(f"Initial {netMixed.metric}: ", initial)
resultMixed = netMixed.Learn(inputs, targets, numEpochs=200)
print(f"Final {netMixed.metric}: ", resultMixed[-1])
plt.plot(np.concatenate((np.array((initial,)), resultMixed)), label="Mixed")


plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()