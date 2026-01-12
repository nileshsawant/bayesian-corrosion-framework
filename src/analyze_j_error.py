#!/usr/bin/env python3
"""
Detailed analysis of why BNN current density predictions are off.
Compare phi field accuracy vs current density accuracy.
"""
import pickle
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from bnn_model import BNNWrapper

print("=" * 80)
print("Analyzing Current Density Error Amplification")
print("=" * 80)

# Load data
data = pickle.load(open('/projects/hpcapps/nsawant/corrosion/bayesian-corrosion-framework/training_data.pkl', 'rb'))
bnn = BNNWrapper.load('/projects/hpcapps/nsawant/corrosion/bayesian-corrosion-framework/bnn_model.pt')

# Parameters for calculation
kappa = 5.0
dy = 0.05

# Test one sample
sample_idx = 0
params = data['inputs'][sample_idx]
phi_true_flat = data['outputs'][sample_idx]
metadata = data['metadata'][sample_idx]

phi_true_2d = phi_true_flat.reshape(metadata['phi_shape'])

# Get BNN prediction
X = torch.tensor([params]).float()
phi_pred_flat, phi_std_flat = bnn.predict(X)
phi_pred_flat = phi_pred_flat.flatten().cpu().numpy()
phi_pred_2d = phi_pred_flat.reshape(metadata['phi_shape'])

print(f"\nSample {sample_idx}: {metadata['params']}")
print(f"Phi field shape: {phi_pred_2d.shape}")

# Extract anode region
phi_surf_true = phi_true_2d[60:120, 0]
phi_abov_true = phi_true_2d[60:120, 1]
phi_surf_pred = phi_pred_2d[60:120, 0]
phi_abov_pred = phi_pred_2d[60:120, 1]

print(f"\nAnode surface phi:")
print(f"  True:  min={phi_surf_true.min():.4f}, max={phi_surf_true.max():.4f}, mean={phi_surf_true.mean():.4f}")
print(f"  Pred:  min={phi_surf_pred.min():.4f}, max={phi_surf_pred.max():.4f}, mean={phi_surf_pred.mean():.4f}")
print(f"  Error: MAE={np.mean(np.abs(phi_surf_true - phi_surf_pred)):.6f} V")

print(f"\nAnode phi above surface:")
print(f"  True:  min={phi_abov_true.min():.4f}, max={phi_abov_true.max():.4f}, mean={phi_abov_true.mean():.4f}")
print(f"  Pred:  min={phi_abov_pred.min():.4f}, max={phi_abov_pred.max():.4f}, mean={phi_abov_pred.mean():.4f}")
print(f"  Error: MAE={np.mean(np.abs(phi_abov_true - phi_abov_pred)):.6f} V")

# Calculate gradients
dphi_true = phi_surf_true - phi_abov_true
dphi_pred = phi_surf_pred - phi_abov_pred

print(f"\nPotential gradient (phi_surface - phi_above):")
print(f"  True:  min={dphi_true.min():.6f}, max={dphi_true.max():.6f}, mean={dphi_true.mean():.6f}")
print(f"  Pred:  min={dphi_pred.min():.6f}, max={dphi_pred.max():.6f}, mean={dphi_pred.mean():.6f}")
print(f"  Error: MAE={np.mean(np.abs(dphi_true - dphi_pred)):.6f} V")
print(f"  Relative error in gradient: {np.mean(np.abs(dphi_true - dphi_pred) / (np.abs(dphi_true) + 1e-10)) * 100:.2f}%")

# Current densities
j_true = kappa * dphi_true / dy
j_pred = kappa * dphi_pred / dy

print(f"\nCurrent Density (J = kappa * dphi / dy):")
print(f"  True:  mean={j_true.mean():.6e} A/m²")
print(f"  Pred:  mean={j_pred.mean():.6e} A/m²")
print(f"  Error: {abs(j_true.mean() - j_pred.mean()):.6e} A/m²")
print(f"  Relative error: {abs(j_true.mean() - j_pred.mean()) / abs(j_true.mean()) * 100:.2f}%")

print(f"\nProfile comparison (first 10 positions):")
for i in range(10):
    err_pct = abs(j_true[i] - j_pred[i]) / (abs(j_true[i]) + 1e-10) * 100
    print(f"  pos {i:2d}: True={j_true[i]:10.6f}, Pred={j_pred[i]:10.6f}, Error={err_pct:6.1f}%")

# Error amplification analysis
print("\n" + "=" * 80)
print("ERROR AMPLIFICATION ANALYSIS")
print("=" * 80)

phi_error_mae = np.mean(np.abs(phi_surf_true - phi_surf_pred))
dphi_error_mae = np.mean(np.abs(dphi_true - dphi_pred))
j_error_mae = np.mean(np.abs(j_true - j_pred))

print(f"Mean Absolute Errors:")
print(f"  Phi field (surface):      {phi_error_mae:.6f} V")
print(f"  Gradient (dphi/dy):       {dphi_error_mae:.6f} V")
print(f"  Current density:          {j_error_mae:.6f} A/m²")
print(f"\nAmplification factor (gradient/phi): {dphi_error_mae / phi_error_mae:.2f}x")
print(f"Amplification factor (J/phi):        {j_error_mae / (phi_error_mae * kappa / dy):.2f}x")

print(f"\nConclusion:")
print(f"  - BNN predicts phi with ~{phi_error_mae:.4f}V error")
print(f"  - Numerical gradient amplifies this to ~{dphi_error_mae:.6f}V error")
print(f"  - Multiplication by kappa/dy = {kappa/dy} gives ~{j_error_mae:.4f} A/m² error")
print(f"  - Since true J is ~{abs(j_true.mean()):.4f} A/m², relative error is ~{abs(j_true.mean() - j_pred.mean()) / abs(j_true.mean()) * 100:.1f}%")

print("\n" + "=" * 80)
