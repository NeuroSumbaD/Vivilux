"""
Test to verify that minus and plus phases produce different activities.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.nets import Net


def test_phase_differences():
    """Test that minus and plus phases produce different activities."""
    print("=== TESTING PHASE DIFFERENCES ===")
    
    # Create network
    net = Net(seed=0)
    
    # Add layers
    input_layer = Layer(2, activation="ReLu", learningRule="GeneRec")
    output_layer = Layer(1, activation="ReLu", learningRule="GeneRec")
    
    net.AddLayer(input_layer)
    net.AddLayer(output_layer)
    
    # Add connection between layers
    net.AddConnection(input_layer, output_layer)
    
    # Test data
    input_data = jnp.array([0.5, 0.8])
    target_data = jnp.array([0.9])
    
    print(f"Input data: {input_data}")
    print(f"Target data: {target_data}")
    
    # Run minus phase only
    print("\n--- MINUS PHASE ONLY ---")
    net.StepPhase("minus", input=input_data)
    minus_input_act = input_layer.getActivity().copy()
    minus_output_act = output_layer.getActivity().copy()
    print(f"Minus phase - Input activity: {minus_input_act}")
    print(f"Minus phase - Output activity: {minus_output_act}")
    
    # Reset and run plus phase only
    print("\n--- PLUS PHASE ONLY ---")
    net.resetActivity()
    net.StepPhase("plus", input=input_data, target=target_data)
    plus_input_act = input_layer.getActivity().copy()
    plus_output_act = output_layer.getActivity().copy()
    print(f"Plus phase - Input activity: {plus_input_act}")
    print(f"Plus phase - Output activity: {plus_output_act}")
    
    # Check if activities are different
    input_diff = jnp.abs(plus_input_act - minus_input_act).max()
    output_diff = jnp.abs(plus_output_act - minus_output_act).max()
    
    print(f"\nInput activity difference: {input_diff}")
    print(f"Output activity difference: {output_diff}")
    
    if output_diff > 1e-6:
        print("✓ Plus and minus phases produce different output activities")
    else:
        print("✗ Plus and minus phases produce the same output activities")
        
    # Run complete trial and check phase history
    print("\n--- COMPLETE TRIAL ---")
    net.resetActivity()
    net.StepTrial("Learn", input=input_data, target=target_data)
    
    print(f"Input layer phaseHist: {input_layer.phaseHist}")
    print(f"Output layer phaseHist: {output_layer.phaseHist}")
    
    if "minus" in output_layer.phaseHist and "plus" in output_layer.phaseHist:
        minus_hist = output_layer.phaseHist["minus"]
        plus_hist = output_layer.phaseHist["plus"]
        hist_diff = jnp.abs(plus_hist - minus_hist).max()
        print(f"Phase history difference: {hist_diff}")
        
        if hist_diff > 1e-6:
            print("✓ Phase history shows different activities")
        else:
            print("✗ Phase history shows same activities")
    else:
        print("✗ Phase history not properly recorded")


if __name__ == "__main__":
    test_phase_differences() 