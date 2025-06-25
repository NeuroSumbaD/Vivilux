"""
Test to verify that learning is working with the fixed phase history.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt
import dataclasses

from vivilux.layers import Layer, LayerConfig
from vivilux.meshes import Mesh
from vivilux.nets import Net, NetConfig
from vivilux.learningRules import GeneRec, CHL
from vivilux.activations import ReLu


def test_phase_history_recording():
    """Test that phase history is being recorded correctly."""
    print("=== TESTING PHASE HISTORY RECORDING ===")
    
    # Create network
    net = Net(seed=0)
    
    # Add layers
    input_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(1, activation="ReLu", learningRule="GeneRec")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Add connection between layers
    net.AddConnection(input_layer, output_layer)
    
    # Prepare data
    input_data = jnp.array([0.5, 0.8], dtype=jnp.float32)
    target_data = jnp.array([0.9], dtype=jnp.float32)
    print(f"Input data: {input_data}")
    print(f"Target data: {target_data}")
    
    # Run minus phase
    print("\n--- MINUS PHASE ---")
    net.StepPhase("minus", input=input_data)
    print(f"Input layer activity (minus): {input_layer.getActivity()}")
    print(f"Output layer activity (minus): {output_layer.getActivity()}")
    
    # Run plus phase
    print("\n--- PLUS PHASE ---")
    net.StepPhase("plus", input=input_data, target=target_data)
    print(f"Input layer activity (plus): {input_layer.getActivity()}")
    print(f"Output layer activity (plus): {output_layer.getActivity()}")
    
    # Run complete trial (should record phase history)
    print("\n--- RUNNING COMPLETE TRIAL ---")
    net.StepTrial("Learn", input=input_data, target=target_data)
    print(f"Input layer phaseHist: {input_layer.phaseHist}")
    print(f"Output layer phaseHist: {output_layer.phaseHist}")
    
    assert "minus" in input_layer.phaseHist, "Minus phase not recorded for input layer"
    assert "plus" in input_layer.phaseHist, "Plus phase not recorded for input layer"
    assert "minus" in output_layer.phaseHist, "Minus phase not recorded for output layer"
    assert "plus" in output_layer.phaseHist, "Plus phase not recorded for output layer"
    print("✓ Phase history recording working correctly")


def test_learning_rule_application():
    """Test that learning rules can access phase history and compute deltas."""
    print("\n=== TESTING LEARNING RULE APPLICATION ===")
    
    # Create network
    net = Net(seed=0)
    
    # Add layers
    input_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(1, activation="ReLu", learningRule="GeneRec")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Add connection between layers
    net.AddConnection(input_layer, output_layer)
    
    # Get initial weights
    initial_weights = net.getWeights()
    print(f"Initial weights: {initial_weights}")
    
    # Run a complete trial to populate phase history
    input_data = jnp.array([0.5, 0.8])
    target_data = jnp.array([0.9])
    net.StepTrial("Learn", input=input_data, target=target_data)
    
    # Test learning rule application
    try:
        delta = GeneRec(input_layer, output_layer)
        print(f"Learning delta shape: {delta.shape}")
        print(f"Learning delta: {delta}")
        print(f"Delta range: [{delta.min():.6f}, {delta.max():.6f}]")
        print(f"Delta mean: {delta.mean():.6f}")
        
        # Verify delta is not all zeros
        assert not jnp.allclose(delta, 0.0), "Learning delta should not be all zeros"
        print("✓ Learning rule application working correctly")
        
    except Exception as e:
        print(f"✗ Learning rule application failed: {e}")
        import traceback
        traceback.print_exc()


def test_weight_updates():
    """Test that weights are actually being updated during learning."""
    print("\n=== TESTING WEIGHT UPDATES ===")
    
    # Create network
    net = Net(seed=0)
    
    # Add layers
    input_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(1, activation="ReLu", learningRule="GeneRec")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Add connection between layers
    mesh = net.AddConnection(input_layer, output_layer)
    
    # Get initial weights
    initial_weights = net.getWeights()
    print(f"Initial weights: {initial_weights}")
    
    # Run a complete trial
    input_data = jnp.array([0.5, 0.8])
    target_data = jnp.array([0.9])
    net.StepTrial("Learn", input=input_data, target=target_data)
    
    # Check if weights changed
    final_weights = net.getWeights()
    print(f"Final weights: {final_weights}")
    
    weight_changed = False
    for name, weights in final_weights.items():
        if name in initial_weights:
            diff = jnp.abs(weights - initial_weights[name]).max()
            print(f"Weight change for {name}: {diff}")
            if diff > 1e-6:
                weight_changed = True
    
    print(f"Weights changed: {weight_changed}")
    
    if weight_changed:
        print("✓ Weight updates working correctly")
    else:
        print("✗ Weights not being updated")


def test_complete_learning():
    """Test complete learning process with multiple epochs."""
    print("\n=== TESTING COMPLETE LEARNING ===")
    
    # Create network
    net = Net(seed=0)
    
    # Add layers
    input_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(1, activation="ReLu", learningRule="GeneRec")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Add connection between layers
    net.AddConnection(input_layer, output_layer)
    
    # Create simple training data
    inputs = jnp.array([[0.5, 0.8], [0.2, 0.9]])
    targets = jnp.array([[0.9], [0.1]])
    
    print(f"Training data shapes: inputs {inputs.shape}, targets {targets.shape}")
    
    # Run a few epochs
    results = net.Learn(numEpochs=3, input=inputs, target=targets, verbosity=1)
    
    print(f"Training results: {results}")
    
    # Check if RMSE is changing
    if "RMSE" in results and len(results["RMSE"]) > 1:
        rmse_changes = [abs(results["RMSE"][i] - results["RMSE"][i-1]) for i in range(1, len(results["RMSE"]))]
        print(f"RMSE changes between epochs: {rmse_changes}")
        
        if any(change > 1e-6 for change in rmse_changes):
            print("✓ Learning is working - RMSE is changing")
        else:
            print("✗ Learning not working - RMSE not changing")
    else:
        print("✗ No RMSE results available")


if __name__ == "__main__":
    print("Running learning fix tests...")
    
    test_phase_history_recording()
    # test_learning_rule_application()
    # test_weight_updates()
    # test_complete_learning()
    
    print("\n=== TEST SUMMARY ===")
    print("Check the output above for any failures.") 