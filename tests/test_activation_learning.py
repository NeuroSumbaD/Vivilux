#!/usr/bin/env python3
"""
Test activation functions and learning rules to verify they work correctly.
This test will help identify why the network is not learning.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from vivilux.activations import (
    create_noisy_xx1_state, noisy_xx1, 
    create_xx1_state, xx1,
    create_sigmoid_state, sigmoid
)
from vivilux.learningRules import CHL, XCAL, GeneRec
from vivilux.layers import Layer, LayerConfig
from vivilux.meshes import Mesh
from vivilux.nets import Net
from vivilux.metrics import RMSE, SSE
import matplotlib.pyplot as plt

def test_activations():
    """Test that activation functions produce reasonable outputs."""
    print("=== Testing Activation Functions ===")
    
    # Test input range
    x = jnp.linspace(-2, 2, 100)
    
    # Test NoisyXX1 (default activation)
    noisy_state = create_noisy_xx1_state()
    noisy_output = noisy_xx1(x, noisy_state)
    print(f"NoisyXX1 output range: [{noisy_output.min():.4f}, {noisy_output.max():.4f}]")
    print(f"NoisyXX1 output at x=0: {noisy_xx1(jnp.array([0.0]), noisy_state)[0]:.4f}")
    print(f"NoisyXX1 output at x=1: {noisy_xx1(jnp.array([1.0]), noisy_state)[0]:.4f}")
    
    # Test XX1 for comparison
    xx1_state = create_xx1_state()
    xx1_output = xx1(x, xx1_state)
    print(f"XX1 output range: [{xx1_output.min():.4f}, {xx1_output.max():.4f}]")
    print(f"XX1 output at x=0: {xx1(jnp.array([0.0]), xx1_state)[0]:.4f}")
    print(f"XX1 output at x=1: {xx1(jnp.array([1.0]), xx1_state)[0]:.4f}")
    
    # Test Sigmoid for comparison
    sigmoid_state = create_sigmoid_state()
    sigmoid_output = sigmoid(x, sigmoid_state)
    print(f"Sigmoid output range: [{sigmoid_output.min():.4f}, {sigmoid_output.max():.4f}]")
    
    # Plot activations
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(x, noisy_output, label='NoisyXX1')
    plt.title('NoisyXX1 Activation')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(x, xx1_output, label='XX1')
    plt.title('XX1 Activation')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(x, sigmoid_output, label='Sigmoid')
    plt.title('Sigmoid Activation')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('activation_tests.png')
    plt.close()
    
    return noisy_output, xx1_output, sigmoid_output

def test_learning_rules():
    """Test that learning rules produce reasonable weight updates."""
    print("\n=== Testing Learning Rules ===")
    
    # Create mock layers with phase history
    class MockLayer:
        def __init__(self, size, name):
            self.size = size
            self.name = name
            # Create realistic phase history
            key = jrandom.PRNGKey(42)
            key1, key2 = jrandom.split(key)
            self.phaseHist = {
                "plus": jrandom.uniform(key1, (size,), minval=0, maxval=1),
                "minus": jrandom.uniform(key2, (size,), minval=0, maxval=1)
            }
        
        def __len__(self):
            return self.size
    
    # Create mock layers
    input_layer = MockLayer(4, "input")
    output_layer = MockLayer(3, "output")
    
    print(f"Input layer plus phase: {input_layer.phaseHist['plus']}")
    print(f"Input layer minus phase: {input_layer.phaseHist['minus']}")
    print(f"Output layer plus phase: {output_layer.phaseHist['plus']}")
    print(f"Output layer minus phase: {output_layer.phaseHist['minus']}")
    
    # Test CHL (default learning rule)
    chl_delta = CHL(input_layer, output_layer)
    print(f"CHL delta shape: {chl_delta.shape}")
    print(f"CHL delta range: [{chl_delta.min():.4f}, {chl_delta.max():.4f}]")
    print(f"CHL delta mean: {chl_delta.mean():.4f}")
    print(f"CHL delta:\n{chl_delta}")
    
    # Test XCAL
    xcal_delta = XCAL(input_layer, output_layer)
    print(f"XCAL delta shape: {xcal_delta.shape}")
    print(f"XCAL delta range: [{xcal_delta.min():.4f}, {xcal_delta.max():.4f}]")
    print(f"XCAL delta mean: {xcal_delta.mean():.4f}")
    print(f"XCAL delta:\n{xcal_delta}")
    
    # Test GeneRec
    generec_delta = GeneRec(input_layer, output_layer)
    print(f"GeneRec delta shape: {generec_delta.shape}")
    print(f"GeneRec delta range: [{generec_delta.min():.4f}, {generec_delta.max():.4f}]")
    print(f"GeneRec delta mean: {generec_delta.mean():.4f}")
    print(f"GeneRec delta:\n{generec_delta}")
    
    return chl_delta, xcal_delta, generec_delta

def test_simple_network():
    """Test a simple network to see if learning is working."""
    print("\n=== Testing Simple Network ===")
    
    # Create a simple 2-layer network
    net = Net(name="test_net")
    
    # Create layers with explicit learning rule
    input_layer = Layer(2, name="input", isInput=True, learningRule="CHL")
    output_layer = Layer(1, name="output", learningRule="CHL")
    
    net.AddLayers([input_layer, output_layer])
    net.AddConnection(input_layer, output_layer)
    
    # Create simple training data
    inputs = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = jnp.array([[0.0], [1.0], [1.0], [0.0]])  # XOR-like pattern
    
    print(f"Training data shapes: inputs {inputs.shape}, targets {targets.shape}")
    
    # Run a few epochs and track metrics
    results = net.Learn(numEpochs=10, input=inputs, target=targets, verbosity=1)
    
    print(f"Training results: {results}")
    
    # Check if weights changed
    initial_weights = net.getWeights()
    print(f"Initial weights: {initial_weights}")
    
    return results, initial_weights

def test_activation_gradients():
    """Test that activation functions have non-zero gradients."""
    print("\n=== Testing Activation Gradients ===")
    
    # Test gradient of NoisyXX1 at single points
    def noisy_xx1_fn(x):
        state = create_noisy_xx1_state()
        return noisy_xx1(x, state)
    
    test_points = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    gradients = []
    
    for point in test_points:
        grad_fn = jax.grad(noisy_xx1_fn)
        grad = grad_fn(point)
        gradients.append(grad)
    
    gradients = jnp.array(gradients)
    print(f"Test points: {test_points}")
    print(f"NoisyXX1 gradients: {gradients}")
    print(f"Gradient range: [{gradients.min():.4f}, {gradients.max():.4f}]")
    
    # Test gradient of XX1 at single points
    def xx1_fn(x):
        state = create_xx1_state()
        return xx1(x, state)
    
    gradients_xx1 = []
    for point in test_points:
        grad_fn_xx1 = jax.grad(xx1_fn)
        grad = grad_fn_xx1(point)
        gradients_xx1.append(grad)
    
    gradients_xx1 = jnp.array(gradients_xx1)
    print(f"XX1 gradients: {gradients_xx1}")
    print(f"XX1 gradient range: [{gradients_xx1.min():.4f}, {gradients_xx1.max():.4f}]")
    
    return gradients, gradients_xx1

def test_learning_rule_application():
    """Test that learning rules are actually being applied in the network."""
    print("\n=== Testing Learning Rule Application ===")
    
    # Create a network and check if learning rules are being called
    net = Net(name="debug_net")
    
    # Create layers
    input_layer = Layer(2, name="input", isInput=True)
    output_layer = Layer(1, name="output")
    
    net.AddLayers([input_layer, output_layer])
    mesh = net.AddConnection(input_layer, output_layer)
    
    # Get initial weights
    initial_weights = net.getWeights()
    print(f"Initial mesh weights: {initial_weights}")
    
    # Create simple data
    inputs = jnp.array([[0.5, 0.5]])
    targets = jnp.array([[0.8]])
    
    # Run one epoch and check if weights changed
    print("Running one epoch...")
    net.RunEpoch("Learn", input=inputs, target=targets, verbosity=1)
    
    # Check if phase history was recorded
    print(f"Input layer phaseHist: {input_layer.state.phaseHist}")
    print(f"Output layer phaseHist: {output_layer.state.phaseHist}")
    
    # Check if weights changed
    final_weights = net.getWeights()
    print(f"Final mesh weights: {final_weights}")
    
    weight_changed = False
    for name, weights in final_weights.items():
        if name in initial_weights:
            diff = jnp.abs(weights - initial_weights[name]).max()
            print(f"Weight change for {name}: {diff}")
            if diff > 1e-6:
                weight_changed = True
    
    print(f"Weights changed: {weight_changed}")
    
    return weight_changed

if __name__ == "__main__":
    print("Running comprehensive activation and learning tests...")
    
    # Run all tests
    test_activations()
    test_learning_rules()
    test_activation_gradients()
    test_learning_rule_application()
    test_simple_network()
    
    print("\n=== Test Summary ===")
    print("If activations have reasonable outputs and learning rules produce non-zero deltas,")
    print("but the network weights don't change, then the issue is in the mesh update logic.")
    print("Check that learning rules are properly connected to mesh weight updates.") 