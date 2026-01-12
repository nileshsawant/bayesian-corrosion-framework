#!/usr/bin/env python3
"""
Test script to validate current density calculation by comparing:
1. Direct output from physics solver (currentDensityProfile)
2. Recalculated from stored phi field
"""
import sys
import os
import numpy as np

# Add the corrosion-modeling-applications to path
sys.path.insert(0, os.path.abspath('/projects/hpcapps/nsawant/corrosion/corrosion-modeling-applications'))

from physics_bridge import PhysicsBridge

print("=" * 80)
print("Current Density Validation Test")
print("=" * 80)

# Initialize physics bridge
repo_path = "/projects/hpcapps/nsawant/corrosion/corrosion-modeling-applications"
physics = PhysicsBridge(repo_path)

# Test with one set of parameters
params = [0.6, 298.0, 8.0, 1.0]  # [NaCl, Temp, pH, Flow]
print(f"\nTest Parameters: NaCl={params[0]}M, T={params[1]}K, pH={params[2]}, Flow={params[3]}m/s")

print("\nRunning physics simulation...")
result = physics.run_sim(params)

if result is None or 'phi' not in result:
    print("ERROR: Simulation failed")
    sys.exit(1)

print("✓ Simulation complete")

# Extract data
phi = result['phi']
print(f"\nPhi field shape: {phi.shape}")

# Check if currentDensity was returned
if 'currentDensity' in result:
    j_direct = result['currentDensity']
    print(f"✓ Direct current density profile: length={len(j_direct)}")
    print(f"  Mean: {np.mean(j_direct):.6e} A/m²")
    print(f"  Min:  {np.min(j_direct):.6e} A/m²")
    print(f"  Max:  {np.max(j_direct):.6e} A/m²")
    print(f"  First 5 values: {j_direct[:5]}")
else:
    print("✗ WARNING: currentDensity not in result!")
    j_direct = None

# Recalculate from phi using the same method as run_physics.m
print("\nRecalculating current density from phi gradient...")

# From run_physics.m:
# cathodeWidth = 3.0m, anodeWidth = 3.0m, deltaDistance = 0.05m
# cathode nodes: 1-60 (MATLAB), anode nodes: 61-120 (MATLAB)
# In Python (0-indexed): cathode 0-59, anode 60-120

cathode_nodes = 60
anode_start = cathode_nodes
anode_end = phi.shape[0]  # Should be 121

print(f"Anode indices: {anode_start} to {anode_end-1} (Python 0-indexed)")

# Extract phi at anode surface and above
# Anode is at bottom (y=0), which is column index 0 in Python
phi_surface = phi[anode_start:anode_end, 0]  # Bottom boundary (y=0)
phi_above = phi[anode_start:anode_end, 1]    # One layer up (y=1)

print(f"phi_surface shape: {phi_surface.shape}")
print(f"phi_surface first 5: {phi_surface[:5]}")
print(f"phi_above first 5: {phi_above[:5]}")

# Calculate current density
# J = kappa * (phi_surface - phi_above) / dy
# From run_physics.m: kappa = 5.0 S/m, dy = gC.aSim.dy
kappa = result.get('conductivity', 5.0)
dy = 0.05  # From run_physics.m: deltaDistance = 0.05

j_recalc = kappa * (phi_surface - phi_above) / dy

print(f"\nRecalculated current density:")
print(f"  Mean: {np.mean(j_recalc):.6e} A/m²")
print(f"  Min:  {np.min(j_recalc):.6e} A/m²")
print(f"  Max:  {np.max(j_recalc):.6e} A/m²")
print(f"  First 5 values: {j_recalc[:5]}")

# Compare if we have both
if j_direct is not None:
    print("\n" + "=" * 80)
    print("COMPARISON: Direct vs Recalculated")
    print("=" * 80)
    
    diff = j_direct - j_recalc
    rel_error = np.abs(diff) / (np.abs(j_direct) + 1e-10)
    
    print(f"Absolute difference:")
    print(f"  Mean: {np.mean(np.abs(diff)):.6e} A/m²")
    print(f"  Max:  {np.max(np.abs(diff)):.6e} A/m²")
    
    print(f"\nRelative error:")
    print(f"  Mean: {np.mean(rel_error)*100:.4f}%")
    print(f"  Max:  {np.max(rel_error)*100:.4f}%")
    
    if np.allclose(j_direct, j_recalc, rtol=1e-5, atol=1e-8):
        print("\n✓✓✓ SUCCESS: Direct and recalculated current densities match!")
    else:
        print("\n✗✗✗ ERROR: Significant mismatch between direct and recalculated values!")
        print(f"\nFirst 10 direct values: {j_direct[:10]}")
        print(f"First 10 recalc values: {j_recalc[:10]}")
        print(f"First 10 differences:   {diff[:10]}")

# Also check corrosion rate
if 'corrosionRate' in result:
    corr_rate_direct = result['corrosionRate']
    corr_rate_recalc = np.mean(j_recalc)
    print(f"\nCorrosion Rate:")
    print(f"  Direct (from physics): {corr_rate_direct:.6e} A/m²")
    print(f"  Recalculated (mean J): {corr_rate_recalc:.6e} A/m²")
    print(f"  Difference: {abs(corr_rate_direct - corr_rate_recalc):.6e} A/m²")

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
