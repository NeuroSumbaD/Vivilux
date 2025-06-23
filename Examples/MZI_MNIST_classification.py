# from CHL_MZI import *
from keras.datasets import mnist
from skimage.measure import block_reduce
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt

numSamples = 40
rngs = nnx.Rngs(0)
(train_X, train_y), (test_X, test_y) = mnist.load_data()

samples = block_reduce(train_X[:numSamples], (1, 4,4), jnp.mean) #truncate for training
samples = samples.reshape(len(samples), -1)/255 # flatten images and normalize

# Convert to JAX arrays
samples = jnp.array(samples)
train_y = jnp.array(train_y)
test_y = jnp.array(test_y)

targets = train_y[:numSamples]
oneHotTargets = jnp.zeros((len(targets), 10))
for index, number in enumerate(targets):
    oneHotTargets = oneHotTargets.at[index, number].set(1)

del train_X, train_y, test_X, test_y

del mnist

inputDimension = samples.shape[1]

# Example: create a stateful RNG for any further randomness
# rngs = nnx.Rngs(0)

for index, thing in enumerate(samples):
    plt.imshow(jnp.array(thing).reshape(7,7))
    plt.title(f'Class: {targets[index]}')
    plt.show()

# Example for model usage (uncomment and adapt as needed):
# from vivilux.nets import Net
# from vivilux.layers import Layer
# from vivilux.meshes import Mesh
# from vivilux.metrics import RMSE
# from copy import deepcopy
# leabraNet = Net(name="MZI_MNIST", seed=0)
# layerList = [Layer(inputDimension, isInput=True, name="Input"),
#              Layer(49, name="Hidden1"),
#              Layer(10, isTarget=True, name="Output")]
# leabraNet.AddLayers(layerList)
# result = leabraNet.Learn(input=samples, target=oneHotTargets, numEpochs=10)
# plt.plot(result["RMSE"])
# plt.show()