#!/usr/bin/env python3
"""
Test script for JAX-optimized processes.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any

# Mock Layer class for testing
class MockLayer:
    def __init__(self, size: int = 10, dtype=jnp.float32):
        self.size = size
        self.dtype = dtype
        self.Ge = jnp.ones(size, dtype=dtype)
        self.Gi_FFFB = jnp.zeros(size, dtype=dtype)
        self.isTarget = False
        self.FFFBparams = {
            "MaxVsAvg": 0.5,
            "FF": 0.1,
            "FF0": 0.0,
            "FBDt": 0.1,
            "FB": 0.1,
            "Gi": 1.0
        }
        self.phaseHist = {
            "plus": jnp.ones(size, dtype=dtype) * 0.5,
            "minus": jnp.ones(size, dtype=dtype) * 0.3
        }
        
        # Mock ActAvg attributes
        self.ActAvg = MockActAvg()
    
    def __len__(self):
        return self.size
    
    def getActivity(self):
        return jnp.random.uniform(0, 1, self.size, dtype=self.dtype)

class MockActAvg:
    def __init__(self):
        self.AvgSLrn = jnp.ones(10, dtype=jnp.float32) * 0.5
        self.AvgM = jnp.ones(10, dtype=jnp.float32) * 0.4
        self.AvgL = jnp.ones(10, dtype=jnp.float32) * 0.6
        self.AvgLLrn = jnp.ones(10, dtype=jnp.float32) * 0.3
        self.AvgS = jnp.ones(10, dtype=jnp.float32) * 0.45

def test_fffb_process():
    """Test FFFB process functionality"""
    print("Testing FFFB process...")
    
    from vivilux.processes import FFFB
    
    # Create mock layer
    layer = MockLayer(size=10)
    
    # Create FFFB process
    fffb = FFFB(layer)
    
    # Initialize state
    state = fffb.AttachLayer(layer)
    
    print(f"Initial state: fbi={state.fbi}, is_floating={state.is_floating}")
    
    # Test StepTime
    for i in range(5):
        state, inhibition = fffb.StepTime(state, layer)
        print(f"Step {i+1}: fbi={state.fbi:.4f}, inhibition={inhibition:.4f}")
    
    # Test UpdateAct
    new_state = fffb.UpdateAct(state, layer)
    print(f"After UpdateAct: pool_act shape={new_state.pool_act.shape}")
    
    # Test Reset
    reset_state = fffb.Reset(layer)
    print(f"After Reset: fbi={reset_state.fbi}")
    
    print("FFFB process test completed successfully!\n")

def test_actavg_process():
    """Test ActAvg process functionality"""
    print("Testing ActAvg process...")
    
    from vivilux.processes import ActAvg
    
    # Create mock layer
    layer = MockLayer(size=10)
    
    # Create ActAvg process
    act_avg = ActAvg(layer, Init=0.15, SSTau=2.0, STau=2.0, MTau=10.0)
    
    # Initialize state
    state = act_avg.AttachLayer(layer)
    
    print(f"Initial state: avg_ss shape={state.avg_ss.shape}, act_p_avg={state.act_p_avg}")
    
    # Test StepTime
    for i in range(5):
        state, _ = act_avg.StepTime(state, layer)
        print(f"Step {i+1}: avg_m mean={jnp.mean(state.avg_m):.4f}")
    
    # Test StepPhase
    state = act_avg.StepPhase(state, layer)
    print(f"After StepPhase: lay_cos_diff_avg={state.lay_cos_diff_avg:.4f}")
    
    # Test InitTrial
    state = act_avg.InitTrial(state, layer)
    print(f"After InitTrial: avg_l_lrn mean={jnp.mean(state.avg_l_lrn):.4f}")
    
    # Test Reset
    reset_state = act_avg.Reset(layer)
    print(f"After Reset: avg_ss mean={jnp.mean(reset_state.avg_ss):.4f}")
    
    print("ActAvg process test completed successfully!\n")

def test_xcal_process():
    """Test XCAL process functionality"""
    print("Testing XCAL process...")
    
    from vivilux.processes import XCAL
    
    # Create mock layers
    snd_layer = MockLayer(size=5)
    rcv_layer = MockLayer(size=8)
    
    # Create XCAL process
    xcal = XCAL(DRev=0.1, DThr=0.0001, Lrate=0.04)
    
    # Initialize state
    state = xcal.AttachLayer(snd_layer, rcv_layer)
    
    print(f"Initial state: norm shape={state.norm.shape}, moment shape={state.moment.shape}")
    
    # Test ErrorDriven
    dwt_error = xcal.ErrorDriven(snd_layer, rcv_layer)
    print(f"ErrorDriven dwt shape={dwt_error.shape}, mean={jnp.mean(dwt_error):.4f}")
    
    # Test BCM
    dwt_bcm = xcal.BCM(snd_layer, rcv_layer)
    print(f"BCM dwt shape={dwt_bcm.shape}, mean={jnp.mean(dwt_bcm):.4f}")
    
    # Test MixedLearn
    dwt_mixed = xcal.MixedLearn(snd_layer, rcv_layer)
    print(f"MixedLearn dwt shape={dwt_mixed.shape}, mean={jnp.mean(dwt_mixed):.4f}")
    
    # Test GetDeltas
    new_state, final_dwt = xcal.GetDeltas(state, snd_layer, rcv_layer)
    print(f"GetDeltas final dwt shape={final_dwt.shape}, mean={jnp.mean(final_dwt):.4f}")
    
    # Test Reset
    reset_state = xcal.Reset(snd_layer, rcv_layer)
    print(f"After Reset: norm mean={jnp.mean(reset_state.norm):.4f}")
    
    print("XCAL process test completed successfully!\n")

def test_jit_compilation():
    """Test that JIT compilation works correctly"""
    print("Testing JIT compilation...")
    
    from vivilux.processes import FFFB, ActAvg, XCAL
    
    # Test FFFB JIT
    layer = MockLayer(size=10)
    fffb = FFFB(layer)
    state = fffb.AttachLayer(layer)
    
    # This should trigger JIT compilation
    print("Compiling FFFB StepTime function...")
    state, inhibition = fffb.StepTime(state, layer)
    print(f"FFFB JIT test passed: inhibition={inhibition:.4f}")
    
    # Test ActAvg JIT
    act_avg = ActAvg(layer)
    state = act_avg.AttachLayer(layer)
    
    print("Compiling ActAvg StepTime function...")
    state, _ = act_avg.StepTime(state, layer)
    print("ActAvg JIT test passed")
    
    # Test XCAL JIT
    snd_layer = MockLayer(size=5)
    rcv_layer = MockLayer(size=8)
    xcal = XCAL()
    state = xcal.AttachLayer(snd_layer, rcv_layer)
    
    print("Compiling XCAL functions...")
    dwt = xcal.ErrorDriven(snd_layer, rcv_layer)
    print(f"XCAL JIT test passed: dwt shape={dwt.shape}")
    
    print("All JIT compilation tests passed!\n")

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing JAX-optimized processes.py")
    print("=" * 50)
    
    try:
        test_fffb_process()
        test_actavg_process()
        test_xcal_process()
        test_jit_compilation()
        
        print("=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 