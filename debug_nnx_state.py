#!/usr/bin/env python3
"""
Debug script to help identify problematic nnx state objects
"""

from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE, ThrMSE, ThrSSE

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt
from copy import deepcopy

def inspect_nnx_state(net):
    """Inspect the nnx state structure to identify problematic objects"""
    print("=== Inspecting nnx state structure ===")
    
    # Get the state structure
    state = nnx.state(net)
    
    print(f"State type: {type(state)}")
    print(f"State length: {len(state)}")
    
    # Inspect each state element
    for i, state_element in enumerate(state):
        print(f"\nState[{i}]:")
        print(f"  Type: {type(state_element)}")
        print(f"  Length: {len(state_element) if hasattr(state_element, '__len__') else 'N/A'}")
        
        if hasattr(state_element, '__len__'):
            for j, sub_element in enumerate(state_element):
                print(f"    State[{i}][{j}]:")
                print(f"      Type: {type(sub_element)}")
                print(f"      Length: {len(sub_element) if hasattr(sub_element, '__len__') else 'N/A'}")
                
                # If it's a list/array, inspect first few elements
                if hasattr(sub_element, '__len__') and len(sub_element) > 0:
                    for k in range(min(5, len(sub_element))):
                        print(f"        State[{i}][{j}][{k}]: {type(sub_element[k])}")
                        print(f"          Value: {sub_element[k]}")
                        if hasattr(sub_element[k], 'shape'):
                            print(f"          Shape: {sub_element[k].shape}")
                        if hasattr(sub_element[k], 'dtype'):
                            print(f"          Dtype: {sub_element[k].dtype}")

def create_test_net():
    """Create a minimal test network similar to randInputs.py"""
    print("Creating test network...")
    
    leabraRunConfig = {
        "DELTA_TIME": 0.001,
        "metrics": {
            "AvgSSE": ThrMSE,
            "SSE": ThrSSE,
            "RMSE": RMSE
        },
        "outputLayers": {
            "target": -1,
        },
        "Learn": ["minus", "plus"],
        "Infer": ["minus"],
    }

    leabraNet = Net(name="LEABRA_NET",
                    runConfig=leabraRunConfig,
                    seed=0)

    # Add layers
    layerList = [Layer(25, isInput=True, name="Input"),
                 Layer(49, name="Hidden1"),
                 Layer(49, name="Hidden2"),
                 Layer(25, isTarget=True, name="Output")]
    
    leabraNet.AddLayers(layerList[:-1])
    outputConfig = deepcopy(layerConfig_std)
    outputConfig["FFFBparams"]["Gi"] = 1.4
    leabraNet.AddLayer(layerList[-1], layerConfig=outputConfig)

    # Add feedforward connections
    ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:])
    
    # Add feedback connections
    fbMeshConfig = {"meshType": Mesh,
                    "meshArgs": {"AbsScale": 1,
                                 "RelScale": 0.2},
                    }
    fbMeshes = leabraNet.AddConnections(layerList[1:], layerList[:-1],
                                        meshConfig=fbMeshConfig)
    
    return leabraNet

def find_problematic_state(net):
    """Find the specific problematic state at [0][34][0]"""
    print("=== Finding problematic state at [0][34][0] ===")
    
    state = nnx.state(net)
    
    try:
        print(f"State[0] type: {type(state[0])}")
        print(f"State[0] length: {len(state[0]) if hasattr(state[0], '__len__') else 'N/A'}")
        
        if hasattr(state[0], '__len__') and len(state[0]) > 34:
            print(f"State[0][34] type: {type(state[0][34])}")
            print(f"State[0][34] length: {len(state[0][34]) if hasattr(state[0][34], '__len__') else 'N/A'}")
            
            if hasattr(state[0][34], '__len__') and len(state[0][34]) > 0:
                print(f"State[0][34][0] type: {type(state[0][34][0])}")
                print(f"State[0][34][0] value: {repr(state[0][34][0])}")
            else:
                print("State[0][34] is empty or not indexable")
        else:
            print("State[0] doesn't have index 34")
            
    except Exception as e:
        print(f"Error accessing state: {e}")

def main():
    print("=== nnx State Debug Script ===")
    
    # Create the test network
    net = create_test_net()
    
    # Find the problematic state
    find_problematic_state(net)
    
    # Inspect the state structure
    inspect_nnx_state(net)
    
    print("\n=== Testing JIT compilation ===")
    
    # Try to identify which method is causing the issue
    try:
        print("Testing StepTrial_jit...")
        # Create some dummy data
        dummy_input = jnp.zeros(25)
        dummy_target = jnp.zeros(25)
        
        # This should work
        net.StepTrial_jit("Learn", input=dummy_input, target=dummy_target)
        print("✓ StepTrial_jit works")
        
    except Exception as e:
        print(f"✗ StepTrial_jit failed: {e}")
        print(f"Error type: {type(e)}")
    
    try:
        print("Testing RunEpoch_jit...")
        # This will likely fail because RunEpoch_jit is not JIT-compatible
        net.RunEpoch_jit("Learn", input=jnp.zeros((1, 25)), target=jnp.zeros((1, 25)))
        print("✓ RunEpoch_jit works")
        
    except Exception as e:
        print(f"✗ RunEpoch_jit failed: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    main() 