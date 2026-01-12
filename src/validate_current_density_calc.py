#!/usr/bin/env python3
"""
Validate current density calculation logic against dataset.
This script loads the training data and recalculates current density 
using the same method as run_physics.m to verify our implementation.
"""
import pickle
import numpy as np

print("=" * 80)
print("Validating Current Density Calculation")
print("=" * 80)

# Load dataset
data = pickle.load(open('/projects/hpcapps/nsawant/corrosion/bayesian-corrosion-framework/training_data.pkl', 'rb'))

print(f"Dataset: {len(data['inputs'])} samples")
print(f"Input shape: {data['inputs'].shape}")
print(f"Output shape: {data['outputs'].shape}")

# Get first sample
sample_idx = 0
phi_true = data['outputs'][sample_idx]
metadata = data['metadata'][sample_idx]

print(f"\nSample {sample_idx}:")
print(f"  Parameters: {metadata['params']}")
print(f"  Phi shape: {metadata['phi_shape']}")
print(f"  Corrosion rate (stored): {metadata['corrosion_rate']:.6e} A/m²")

# Reshape phi
phi_shape = metadata['phi_shape']
phi_2d = phi_true.reshape(phi_shape)
print(f"  Reshaped phi: {phi_2d.shape}")

# From run_physics.m:
# Line 9-10: cathodeWidth = 3.0m, anodeWidth = 3.0m
# Line 12: deltaDistance = 0.05m
# Line 14: aNodes = round(3.0/0.05) = 60
# Line 15: cNodes = round(3.0/0.05) = 60
# Line 50-52: startIdx = cNodes + 1 = 61 (MATLAB 1-indexed)
#             endIdx = cNodes + aNodes = 120
#             anodeIndices = 61:120 (MATLAB)

# In Python (0-indexed):
# Cathode: indices 0-59 (60 nodes)
# Anode: indices 60-119 (60 nodes)  
# Total: 0-120 (121 nodes)

cathode_nodes = 60
anode_nodes = 60
anode_start = cathode_nodes  # 60
anode_end = anode_start + anode_nodes  # 120 (exclusive in slice)

print(f"\nGeometry (Python 0-indexed):")
print(f"  Cathode nodes: 0 to {cathode_nodes-1} ({cathode_nodes} total)")
print(f"  Anode nodes: {anode_start} to {anode_end-1} ({anode_nodes} total)")
print(f"  Total nodes: {phi_shape[0]}")

# Extract phi at anode
# MATLAB: phi(anodeIndices, 1) means rows 61-120, column 1
# Python: phi[60:120, 0] means rows 60-119, column 0

phi_surface = phi_2d[anode_start:anode_end, 0]  # Bottom (y=0) - column index 0
phi_above = phi_2d[anode_start:anode_end, 1]    # One up (y=1) - column index 1

print(f"\nPhi extraction:")
print(f"  phi_surface shape: {phi_surface.shape} (should be ({anode_nodes},))")
print(f"  phi_above shape: {phi_above.shape}")
print(f"  phi_surface range: [{phi_surface.min():.4f}, {phi_surface.max():.4f}] V")
print(f"  phi_above range: [{phi_above.min():.4f}, {phi_above.max():.4f}] V")
print(f"  Difference range: [{(phi_surface - phi_above).min():.4e}, {(phi_surface - phi_above).max():.4e}] V")

# Calculate current density
# From run_physics.m line 86: currentDensityProfile = kappa * (phi_surface - phi_above) / dy
kappa = 5.0  # S/m (line 73 of run_physics.m)
dy = 0.05    # meters (deltaDistance from line 12)

j_calc = kappa * (phi_surface - phi_above) / dy

print(f"\nCurrent Density Calculation:")
print(f"  kappa = {kappa} S/m")
print(f"  dy = {dy} m")
print(f"  Formula: J = kappa * (phi_surface - phi_above) / dy")

print(f"\nResults:")
print(f"  Current density range: [{j_calc.min():.4e}, {j_calc.max():.4e}] A/m²")
print(f"  Mean (corrosion rate): {j_calc.mean():.6e} A/m²")
print(f"  Stored corrosion rate: {metadata['corrosion_rate']:.6e} A/m²")

# Compare
diff = abs(j_calc.mean() - metadata['corrosion_rate'])
rel_error = diff / abs(metadata['corrosion_rate']) * 100

print(f"\nValidation:")
print(f"  Absolute difference: {diff:.6e} A/m²")
print(f"  Relative error: {rel_error:.4f}%")

if rel_error < 1.0:  # Accept < 1% error (numerical precision)
    print("  ✓✓✓ PASS: Calculation matches stored value!")
else:
    print("  ✗✗✗ FAIL: Significant mismatch!")

# Show profile values
print(f"\nCurrent density profile (first 10 values):")
for i in range(min(10, len(j_calc))):
    print(f"  j[{i}] = {j_calc[i]:.6e} A/m²")

# Statistical analysis
print(f"\nStatistics:")
print(f"  Mean: {np.mean(j_calc):.6e} A/m²")
print(f"  Std:  {np.std(j_calc):.6e} A/m²")
print(f"  Min:  {np.min(j_calc):.6e} A/m²")
print(f"  Max:  {np.max(j_calc):.6e} A/m²")

print("\n" + "=" * 80)
print("Testing All Samples")
print("=" * 80)

errors = []
for idx in range(len(data['inputs'])):
    phi_flat = data['outputs'][idx]
    meta = data['metadata'][idx]
    phi_2d = phi_flat.reshape(meta['phi_shape'])
    
    # Extract anode region
    phi_surf = phi_2d[60:120, 0]
    phi_abov = phi_2d[60:120, 1]
    
    # Calculate
    j = kappa * (phi_surf - phi_abov) / dy
    corr_rate_calc = np.mean(j)
    corr_rate_stored = meta['corrosion_rate']
    
    rel_err = abs(corr_rate_calc - corr_rate_stored) / abs(corr_rate_stored) * 100
    errors.append(rel_err)
    
    status = "✓" if rel_err < 1.0 else "✗"
    print(f"  Sample {idx:2d}: Calc={corr_rate_calc:.6e}, Stored={corr_rate_stored:.6e}, Error={rel_err:.3f}% {status}")

print(f"\nSummary:")
print(f"  Mean error: {np.mean(errors):.4f}%")
print(f"  Max error:  {np.max(errors):.4f}%")
print(f"  Samples passing (<1%): {np.sum(np.array(errors) < 1.0)}/{len(errors)}")

print("\n" + "=" * 80)
print("Validation Complete")
print("=" * 80)
