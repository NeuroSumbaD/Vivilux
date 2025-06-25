"""
Debug test to identify why network training is not working.
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
from vivilux.learningRules import GeneRec
from vivilux.activations import ReLu


def debug_network_state_flow():
    """Debug the network state flow during phases."""
    print("=== DEBUGGING NETWORK STATE FLOW ===")
    
    # Create network
    net = Net(seed=0)  # Use seed instead of rngs
    
    # Add layers
    input_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(1, activation="ReLu", learningRule="GeneRec")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Add connection between layers
    net.AddConnection(input_layer, output_layer)
    
    print(f"Network state after creation: {net.state is not None}")
    if net.state:
        print(f"Number of layer states: {len(net.state.layerStates)}")
    
    # Test with input data
    input_data = jnp.array([0.5, 0.8])
    print(f"\nInput data: {input_data}")
    
    # Run plus phase
    print("\n--- PLUS PHASE ---")
    net.StepPhase("plus", input=input_data)
    
    if net.state:
        print(f"Input layer activity: {net.state.layerStates[0].Act}")
        print(f"Output layer activity: {net.state.layerStates[1].Act}")
        print(f"Input layer Vm: {net.state.layerStates[0].Vm}")
        print(f"Output layer Vm: {net.state.layerStates[1].Vm}")
    
    # Run minus phase
    print("\n--- MINUS PHASE ---")
    net.StepPhase("minus", input=input_data)
    
    if net.state:
        print(f"Input layer activity: {net.state.layerStates[0].Act}")
        print(f"Output layer activity: {net.state.layerStates[1].Act}")
    
    # Check phase history
    print(f"\nInput layer phase history: {hasattr(input_layer, 'phaseHist')}")
    if hasattr(input_layer, 'phaseHist'):
        print(f"Plus phase: {input_layer.phaseHist.get('plus', 'None')}")
        print(f"Minus phase: {input_layer.phaseHist.get('minus', 'None')}")


def debug_learning_rule():
    """Debug the learning rule application."""
    print("\n=== DEBUGGING LEARNING RULE ===")
    
    # Create layers with learning rules
    input_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(1, activation="ReLu", learningRule="GeneRec")
    
    # Create mesh and initialize manually
    mesh = Mesh(2, input_layer)
    from vivilux.meshes import create_mesh_state
    mesh.state = create_mesh_state(mesh.config, jrandom.PRNGKey(0))
    
    print(f"Initial mesh weights:\n{mesh.get()}")
    
    # Set up phase history manually
    input_layer.phaseHist = {
        "plus": jnp.array([0.6, 0.7]),
        "minus": jnp.array([0.4, 0.5])
    }
    output_layer.phaseHist = {
        "plus": jnp.array([0.8]),
        "minus": jnp.array([0.3])
    }
    
    print(f"Input plus phase: {input_layer.phaseHist['plus']}")
    print(f"Input minus phase: {input_layer.phaseHist['minus']}")
    print(f"Output plus phase: {output_layer.phaseHist['plus']}")
    print(f"Output minus phase: {output_layer.phaseHist['minus']}")
    
    # Apply learning rule
    try:
        delta = GeneRec(input_layer, output_layer)
        print(f"Learning delta:\n{delta}")
        
        # Apply to mesh - manually set the matrix
        initial_weights = mesh.get().copy()
        # For now, just add a small delta to simulate learning
        new_weights = initial_weights + delta * 0.01  # Small learning rate
        mesh.set(new_weights)
        updated_weights = mesh.get()
        
        print(f"Initial weights:\n{initial_weights}")
        print(f"Updated weights:\n{updated_weights}")
        print(f"Weight change:\n{updated_weights - initial_weights}")
        
    except Exception as e:
        print(f"Error applying learning rule: {e}")
        import traceback
        traceback.print_exc()


def debug_mesh_integration():
    """Debug mesh integration by manually applying mesh weights."""
    print("\n=== DEBUGGING MESH INTEGRATION ===")
    
    # Create network
    net = Net(seed=0)
    
    # Add layers
    input_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(1, activation="ReLu", learningRule="GeneRec")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Add connection between layers
    mesh = net.AddConnection(input_layer, output_layer)
    
    # Initialize mesh state
    mesh.apply()
    
    print(f"Mesh weights:\n{mesh.get()}")
    print(f"Input layer activity: {input_layer.getActivity()}")
    print(f"Output layer activity: {output_layer.getActivity()}")
    
    # Manually apply mesh weights to simulate integration
    input_activity = jnp.array([0.5, 0.8])
    mesh_weights = mesh.get()
    
    # Compute input to output layer: weights * input_activity
    output_input = jnp.dot(mesh_weights, input_activity)
    print(f"Computed output input: {output_input}")
    
    # Apply this input to the output layer's GeRaw
    if net.state:
        # Update output layer's GeRaw with mesh input
        new_layer_states = list(net.state.layerStates)
        output_state = new_layer_states[1]
        new_layer_states[1] = dataclasses.replace(output_state, GeRaw=output_input)
        net.state = dataclasses.replace(net.state, layerStates=new_layer_states)
        
        # Step the output layer to see if it responds
        net.StepPhase("plus", input=input_activity)
        
        print(f"After mesh integration:")
        print(f"Input layer activity: {net.state.layerStates[0].Act}")
        print(f"Output layer activity: {net.state.layerStates[1].Act}")
        print(f"Output layer Vm: {net.state.layerStates[1].Vm}")
        print(f"Output layer GeRaw: {net.state.layerStates[1].GeRaw}")


def debug_epoch_execution():
    """Debug a complete epoch execution."""
    print("\n=== DEBUGGING EPOCH EXECUTION ===")
    
    # Create network
    net = Net(seed=0)  # Use seed instead of rngs
    
    # Add layers
    input_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(1, activation="ReLu", learningRule="GeneRec")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Add connection between layers
    net.AddConnection(input_layer, output_layer)
    
    # Create training data
    inputs = jnp.array([[0.5, 0.8], [0.2, 0.9]])
    targets = jnp.array([[1.0], [0.0]])
    
    print(f"Training inputs: {inputs}")
    print(f"Training targets: {targets}")
    
    # Get initial state
    if net.state:
        initial_output_act = net.state.layerStates[1].Act.copy()
        print(f"Initial output activity: {initial_output_act}")
    
    # Run one epoch
    print("\nRunning epoch...")
    try:
        # Add debug output to see what's happening
        print(f"Before epoch - mesh states: {len(net.state.meshStates) if net.state else 0}")
        if net.state and len(net.state.meshStates) > 0:
            print(f"Mesh weights: {net.state.meshStates[0].matrix}")
        
        net.RunEpoch("Learn", input=inputs, target=targets)
        print("Epoch completed successfully")
        
        if net.state:
            final_output_act = net.state.layerStates[1].Act
            print(f"Final output activity: {final_output_act}")
            print(f"Final output GeRaw: {net.state.layerStates[1].GeRaw}")
            
            # Check if activity changed
            if jnp.allclose(initial_output_act, final_output_act):
                print("⚠️  WARNING: Output activity did not change!")
            else:
                print("✓ Output activity changed")
        
        # Check phase history
        print(f"Input layer has phase history: {hasattr(input_layer, 'phaseHist')}")
        if hasattr(input_layer, 'phaseHist'):
            print(f"Phase history keys: {list(input_layer.phaseHist.keys())}")
        
    except Exception as e:
        print(f"Error during epoch: {e}")
        import traceback
        traceback.print_exc()


def debug_metric_tracking():
    """Debug metric computation and tracking."""
    print("\n=== DEBUGGING METRIC TRACKING ===")
    
    from vivilux.metrics import RMSE, SSE
    
    # Create network
    net = Net(seed=0)  # Use seed instead of rngs
    
    # Add layers
    input_layer = Layer(2, activation="ReLu")
    output_layer = Layer(1, activation="ReLu")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Add connection between layers
    net.AddConnection(input_layer, output_layer)
    
    # Create training data
    inputs = jnp.array([[0.5, 0.8], [0.2, 0.9]])
    targets = jnp.array([[1.0], [0.0]])
    
    # Run a few epochs and track metrics
    print("Running epochs and tracking metrics...")
    
    for epoch in range(3):
        print(f"\n--- EPOCH {epoch + 1} ---")
        
        # Run epoch
        net.RunEpoch("Learn", input=inputs, target=targets)
        
        # Get current predictions
        if net.state:
            predictions = net.state.layerStates[1].Act.reshape(-1, 1)
            print(f"Predictions: {predictions}")
            print(f"Targets: {targets}")
            
            # Compute metrics
            rmse = RMSE(predictions, targets)
            sse = SSE(predictions, targets)
            
            print(f"RMSE: {rmse}")
            print(f"SSE: {sse}")
            
            # Check if metrics are changing
            if epoch > 0:
                if jnp.allclose(rmse, prev_rmse, atol=1e-6):
                    print("⚠️  WARNING: RMSE not changing between epochs!")
                else:
                    print("✓ RMSE is changing")
            
            prev_rmse = rmse


def run_debug_tests():
    """Run all debug tests."""
    print("Running debug tests to identify training issues...\n")
    
    debug_network_state_flow()
    debug_learning_rule()
    debug_mesh_integration()
    debug_epoch_execution()
    debug_metric_tracking()
    
    print("\n=== DEBUG SUMMARY ===")
    print("Check the output above for any warnings or errors.")
    print("Key things to look for:")
    print("1. Are layer activities changing during forward pass?")
    print("2. Are learning rules computing non-zero deltas?")
    print("3. Are weights being updated?")
    print("4. Are metrics changing between epochs?")


if __name__ == "__main__":
    run_debug_tests() 