import numpy as np

def RMSE(predictions, targets):
    '''Square Root of Mean Squared Error'''
    errors = np.sum(np.square(predictions - targets), axis=1)
    return np.sqrt(np.mean(errors))

def MSE(predictions, targets):
    ''' Mean Squared Error'''
    errors = np.sum(np.square(predictions - targets), axis=1)
    return np.mean(errors)

def SSE(predictions, targets):
    '''Sum of Squared Error'''
    errors = np.sum(np.square(predictions - targets), axis=1)
    return np.mean(errors)

def MAE(predictions, targets):
    '''Mean Absolute Error'''
    errors = np.sum(np.abs(predictions - targets), axis=1)
    return np.mean(errors)

def SAE(predictions, targets):
    '''Sum of Absolute Error'''
    errors = np.sum(np.abs(predictions - targets), axis=1)
    return errors

def HardmaxAccuracy(predictions, targets):
    '''Hardmax translates the prediction into a one-hot vector to compare
        against a one-hot encoded classification dataset
    '''
    predictedClass = np.argmax(predictions, axis=1)
    groundTruth = np.argmax(targets)
    accuracy = np.sum((predictedClass == groundTruth).astype("int"))/len(targets)
    return accuracy

def ThrMSE(predictions, targets, tol = 0.5):
    '''Calculates the MSE but with a tolerance for errors. When the absolute
        error is less than `tol` the error is not added to MSE.
    '''
    numNeurons = len(targets[0]) # grab dimension of first vector
    errors = np.array(predictions - targets)
    errors[np.abs(errors) < tol] = 0
    errors = np.sum(np.square(errors), axis=1)
    return np.mean(errors/numNeurons)

def ThrSSE(predictions, targets, tol = 0.5):
    '''Calculates the MSE but with a tolerance for errors. When the absolute
        error is less than `tol` the error is not added to MSE.
    '''
    numNeurons = len(targets[0]) # grab dimension of first vector
    errors = np.array(predictions - targets)
    errors[errors < tol] = 0
    errors = np.sum(np.square(errors), axis=1)
    return np.mean(errors)