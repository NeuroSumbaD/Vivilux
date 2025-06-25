"""
Debug script to check if mesh updates are being called during learning.
"""
from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE, ThrMSE, ThrSSE

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

# Create a simple network
net = Net(seed=0)

# Add layers
input_layer = Layer(2, isInput=True, name="Input")
hidden_layer = Layer(3, name="Hidden", learningRule="CHL")
output_layer = Layer(1, isTarget=True, name="Output")

net.AddLayer(input_layer)
net.AddLayer(hidden_layer)
net.AddLayer(output_layer)

# Add connections
net.AddConnection(input_layer, hidden_layer)
net.AddConnection(hidden_layer, output_layer)

print(f"Input layer excMeshes: {len(input_layer.excMeshes)}")
print(f"Hidden layer excMeshes: {len(hidden_layer.excMeshes)}")
print(f"Output layer excMeshes: {len(output_layer.excMeshes)}")

# Check if meshes have names
for layer in [input_layer, hidden_layer, output_layer]:
    for i, mesh in enumerate(layer.excMeshes):
        print(f"{layer.name} mesh {i}: {mesh.name}")

# Create simple data
input_data = jnp.array([0.5, 0.8])
target_data = jnp.array([0.9])

print("\nRunning a single trial...")
net.StepTrial("Learn", input=input_data, target=target_data)

print("\nChecking phase history...")
print(f"Input layer phaseHist: {input_layer.phaseHist}")
print(f"Hidden layer phaseHist: {hidden_layer.phaseHist}")
print(f"Output layer phaseHist: {output_layer.phaseHist}")

print("\nRunning a single epoch...")
net.RunEpoch("Learn", input=input_data, target=target_data)

print("Done!") 