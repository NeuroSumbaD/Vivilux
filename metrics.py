import numpy as np

def RMSE(predictions, targets):
    errors = np.sum(np.square(predictions - targets), axis=1)
    return np.sqrt(np.mean(errors))

def HardmaxAccuracy(predictions, targets):
    predictedClass = np.argmax(predictions, axis=1)
    groundTruth = np.argmax(targets)
    accuracy = np.sum((predictedClass == groundTruth).astype("int"))/len(targets)
    return accuracy