"""
Test if flow velocity actually affects physics output
by running two simulations directly
"""
import sys
sys.path.insert(0, '.')

from physics_bridge import PhysicsBridge

# Initialize bridge
dep_path = '../../corrosion-modeling-applications'
bridge = PhysicsBridge(dep_path)

# Test parameters
params1 = [0.1, 278.0, 6.0, 0.1]   # Low flow
params2 = [0.1, 278.0, 6.0, 3.0]   # High flow

print("Running simulation with FLOW = 0.1 m/s...")
result1 = bridge.run_sim(params1)

print("\nRunning simulation with FLOW = 3.0 m/s...")
result2 = bridge.run_sim(params2)

if result1 is not None and result2 is not None:
    import numpy as np
    
    phi1 = result1['phi']
    phi2 = result2['phi']
    
    diff = phi2 - phi1
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Flow 0.1 m/s: phi shape={phi1.shape}, mean={np.mean(phi1):.6e}, std={np.std(phi1):.6e}")
    print(f"Flow 3.0 m/s: phi shape={phi2.shape}, mean={np.mean(phi2):.6e}, std={np.std(phi2):.6e}")
    print(f"\nDifference:")
    print(f"  Mean: {np.mean(diff):.6e}")
    print(f"  Std:  {np.std(diff):.6e}")
    print(f"  Max:  {np.max(np.abs(diff)):.6e}")
    print(f"  Min:  {np.min(np.abs(diff)):.6e}")
    
    if np.max(np.abs(diff)) < 1e-10:
        print("\n⚠️  OUTPUTS ARE IDENTICAL!")
        print("Flow velocity is NOT affecting the simulation.")
    else:
        print("\n✓ Outputs differ - flow velocity is working!")
else:
    print("\n✗ Simulation failed!")
