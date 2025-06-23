import jax.numpy as jnp
from jax import jit

@jit
def RMSE(predictions: jnp.ndarray, targets: jnp.ndarray):
    '''Square Root of Mean Squared Error'''
    errors = jnp.sum(jnp.square(predictions - targets), axis=1)
    return jnp.sqrt(jnp.mean(errors))

@jit
def MSE(predictions: jnp.ndarray, targets: jnp.ndarray):
    ''' Mean Squared Error'''
    errors = jnp.sum(jnp.square(predictions - targets), axis=1)
    return jnp.mean(errors)

@jit
def SSE(predictions: jnp.ndarray, targets: jnp.ndarray):
    '''Sum of Squared Error'''
    errors = jnp.sum(jnp.square(predictions - targets), axis=1)
    return jnp.mean(errors)

@jit
def MAE(predictions: jnp.ndarray, targets: jnp.ndarray):
    '''Mean Absolute Error'''
    errors = jnp.sum(jnp.abs(predictions - targets), axis=1)
    return jnp.mean(errors)

@jit
def SAE(predictions: jnp.ndarray, targets: jnp.ndarray):
    '''Sum of Absolute Error'''
    errors = jnp.sum(jnp.abs(predictions - targets), axis=1)
    return errors

@jit
def HardmaxAccuracy(predictions: jnp.ndarray, targets: jnp.ndarray):
    '''Hardmax translates the prediction into a one-hot vector to compare
        against a one-hot encoded classification dataset
    '''
    predictedClass = jnp.argmax(predictions, axis=1)
    groundTruth = jnp.argmax(targets)
    accuracy = jnp.sum((predictedClass == groundTruth).astype(jnp.int32))/len(targets)
    return accuracy

@jit
def ThrMSE(predictions: jnp.ndarray, targets: jnp.ndarray, tol = 0.5):
    '''Calculates the MSE but with a tolerance for errors. When the absolute
        error is less than `tol` the error is not added to MSE.
    '''
    numNeurons = len(targets[0]) # grab dimension of first vector
    errors = predictions - targets
    errors = jnp.where(jnp.abs(errors) < tol, 0, errors)
    errors = jnp.sum(jnp.square(errors), axis=1)
    return jnp.mean(errors/numNeurons)

@jit
def ThrSSE(predictions: jnp.ndarray, targets: jnp.ndarray, tol = 0.5):
    '''Calculates the MSE but with a tolerance for errors. When the absolute
        error is less than `tol` the error is not added to MSE.
    '''
    numNeurons = len(targets[0]) # grab dimension of first vector
    errors = predictions - targets
    errors = jnp.where(jnp.abs(errors) < tol, 0, errors)
    errors = jnp.sum(jnp.square(errors), axis=1)
    return jnp.mean(errors)