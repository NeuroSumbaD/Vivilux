"""
Test layer state updates and learning functionality.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt
import dataclasses

from vivilux.layers import Layer, LayerConfig
from vivilux.core.layer import update_conductance, step_time, reset_activity
from vivilux.meshes import Mesh, MeshConfig, create_mesh_state
from vivilux.nets import Net, NetConfig
from vivilux.learningRules import GeneRec
from vivilux.activations import ReLu


def test_layer_activation_updates():
    """Test that layer activations change during forward pass."""
    print("Testing layer activation updates...")
    
    # Create a simple layer
    config = LayerConfig(length=5, activation="ReLu")
    layer = Layer(5, activation="ReLu")
    
    # Create initial state
    from vivilux.layers import create_layer_state
    state = create_layer_state(config, jrandom.PRNGKey(0))
    
    # Verify initial state
    print(f"Initial Act: {state.Act}")
    print(f"Initial Vm: {state.Vm}")
    
    # Apply some input (simulate mesh output)
    input_activity = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    state = dataclasses.replace(state, GeRaw=input_activity, Ge=input_activity)
    
    # Update conductances
    dt = 0.1
    key = jrandom.PRNGKey(1)
    new_state = update_conductance(state, config, dt, key)
    
    print(f"After conductance update:")
    print(f"  Ge: {new_state.Ge}")
    print(f"  Gi: {new_state.Gi}")
    
    # Step time to update activations
    time = 0.1
    final_state = step_time(new_state, config, time, dt, key)
    
    print(f"After time step:")
    print(f"  Act: {final_state.Act}")
    print(f"  Vm: {final_state.Vm}")
    
    # Verify that activations changed
    assert not jnp.allclose(state.Act, final_state.Act), "Activations should change during forward pass"
    assert not jnp.allclose(state.Vm, final_state.Vm), "Membrane potential should change"
    
    print("✓ Layer activation updates working correctly")


def test_mesh_weight_updates():
    """Test that mesh weights change during learning."""
    print("\nTesting mesh weight updates...")
    
    # Create layers
    input_layer = Layer(3, activation="ReLu")
    output_layer = Layer(2, activation="ReLu")
    
    # Create mesh connecting them
    mesh = Mesh(3, input_layer)
    # Initialize mesh state manually since layer has no network
    from vivilux.meshes import create_mesh_state
    mesh.state = create_mesh_state(mesh.config, jrandom.PRNGKey(0))
    
    # Get initial weights
    initial_weights = mesh.get().copy()
    print(f"Initial weights:\n{initial_weights}")
    
    # Create some delta (simulate learning rule output) - match the mesh shape (3, 3)
    delta = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    
    # Apply update: manually set new weights
    new_weights = initial_weights + delta * 0.01
    mesh.set(new_weights)
    
    # Get updated weights
    updated_weights = mesh.get()
    print(f"Updated weights:\n{updated_weights}")
    
    # Verify weights changed
    assert not jnp.allclose(initial_weights, updated_weights), "Weights should change during learning"
    
    print("✓ Mesh weight updates working correctly")


def test_network_state_propagation():
    """Test that network state propagates through layers."""
    print("\nTesting network state propagation...")
    
    # Create a simple network
    net = Net(seed=0)
    
    # Add layers
    input_layer = Layer(3, activation="ReLu")
    hidden_layer = Layer(4, activation="ReLu")
    output_layer = Layer(2, activation="ReLu")
    
    net.AddLayer(input_layer)
    net.AddLayer(hidden_layer)
    net.AddLayer(output_layer)
    
    # Create input data
    input_data = jnp.array([0.5, 0.3, 0.8])
    
    # Run a single phase
    net.StepPhase("plus", input=input_data)
    
    # Check that state was created and updated
    if net.state is not None:
        assert len(net.state.layerStates) == 3, "Should have 3 layer states"
        # Check that input layer was clamped
        input_act = net.state.layerStates[0].Act
        print(f"Input layer activity: {input_act}")
        assert not jnp.allclose(input_act, jnp.zeros_like(input_act)), "Input layer should be clamped"
    else:
        raise AssertionError("Network state should be created")
    
    print("✓ Network state propagation working correctly")


def test_learning_rule_application():
    """Test that learning rules compute and apply deltas."""
    print("\nTesting learning rule application...")
    
    # Create layers with learning rules
    input_layer = Layer(3, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    
    # Create mesh with size matching output layer (2) and input layer size (3)
    mesh = Mesh(2, input_layer)  # size=2 to match output layer
    from vivilux.meshes import create_mesh_state
    mesh.state = create_mesh_state(mesh.config, jrandom.PRNGKey(0))
    
    # Simulate phase activities
    input_layer.phaseHist = {
        "plus": jnp.array([0.1, 0.2, 0.3]),
        "minus": jnp.array([0.05, 0.15, 0.25])
    }
    output_layer.phaseHist = {
        "plus": jnp.array([0.4, 0.5]),
        "minus": jnp.array([0.35, 0.45])
    }
    
    # Get initial weights
    initial_weights = mesh.get().copy()
    
    # Apply learning rule
    delta = GeneRec(input_layer, output_layer)
    print(f"Learning delta:\n{delta}")
    
    # Apply delta to mesh: manually set new weights
    new_weights = initial_weights + delta * 0.01
    mesh.set(new_weights)
    
    # Get updated weights
    updated_weights = mesh.get()
    
    # Verify weights changed
    assert not jnp.allclose(initial_weights, updated_weights), "Weights should change after learning rule"
    
    print("✓ Learning rule application working correctly")


def test_metric_computation():
    """Test that metrics reflect actual changes."""
    print("\nTesting metric computation...")
    
    from vivilux.metrics import RMSE, SSE
    
    # Create predictions and targets
    predictions = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    targets = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    
    # Compute metrics
    rmse = RMSE(predictions, targets)
    sse = SSE(predictions, targets)
    
    print(f"RMSE: {rmse}")
    print(f"SSE: {sse}")
    
    # Verify metrics are reasonable
    assert rmse > 0, "RMSE should be positive"
    assert sse > 0, "SSE should be positive"
    
    # Test with perfect predictions
    perfect_predictions = targets.copy()
    perfect_rmse = RMSE(perfect_predictions, targets)
    perfect_sse = SSE(perfect_predictions, targets)
    
    print(f"Perfect RMSE: {perfect_rmse}")
    print(f"Perfect SSE: {perfect_sse}")
    
    assert jnp.allclose(perfect_rmse, 0.0, atol=1e-6), "Perfect predictions should have zero RMSE"
    assert jnp.allclose(perfect_sse, 0.0, atol=1e-6), "Perfect predictions should have zero SSE"
    
    print("✓ Metric computation working correctly")


def test_end_to_end_training():
    """Test complete training loop with verification."""
    print("\nTesting end-to-end training...")
    
    # Create simple network
    net = Net(seed=0)
    
    # Add layers
    input_layer = Layer(3, activation="ReLu")
    output_layer = Layer(2, activation="ReLu")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Create training data
    inputs = jnp.array([[0.5, 0.3, 0.8], [0.2, 0.7, 0.1]])
    targets = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    
    # Get initial state
    if net.state is not None:
        initial_state = net.state.layerStates[1].Act.copy()  # Output layer
    else:
        raise AssertionError("Network state should be created")
    
    # Run a few epochs
    for epoch in range(5):
        net.RunEpoch("Learn", input=inputs, target=targets)
        
        # Check current state
        if net.state is not None:
            current_state = net.state.layerStates[1].Act
            print(f"Epoch {epoch + 1}, Output activity: {current_state}")
            # Verify state is changing
            if epoch > 0:
                assert not jnp.allclose(initial_state, current_state), f"State should change by epoch {epoch + 1}"
        else:
            raise AssertionError("Network state should be created")
    
    print("✓ End-to-end training working correctly")


def test_mesh_creation():
    """Test that Mesh and MeshConfig create a valid mesh state and matrix."""
    config = MeshConfig(size=3, in_layer_size=2)
    key = jrandom.PRNGKey(0)
    state = create_mesh_state(config, key)
    assert state.matrix.shape == (3, 2)
    assert hasattr(state, 'matrix')
    print("✓ Mesh creation and state initialization works")


def run_all_tests():
    """Run all unit tests."""
    print("Running comprehensive unit tests for neural network components...\n")
    
    try:
        test_layer_activation_updates()
        test_mesh_weight_updates()
        test_network_state_propagation()
        test_learning_rule_application()
        test_metric_computation()
        test_end_to_end_training()
        
        print("\n🎉 All tests passed! Neural network components are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
    test_mesh_creation() 