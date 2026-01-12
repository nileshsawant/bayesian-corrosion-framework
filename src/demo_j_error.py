#!/usr/bin/env python3
"""
Quick demonstration of the current density error issue.
Shows that BNN predicts phi accurately but J derivation fails.
"""
import pickle
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from bnn_model import BNNWrapper

# Load data
data = pickle.load(open('/projects/hpcapps/nsawant/corrosion/bayesian-corrosion-framework/training_data.pkl', 'rb'))
bnn = BNNWrapper.load('/projects/hpcapps/nsawant/corrosion/bayesian-corrosion-framework/bnn_model.pt')

# Test sample
idx = 0
params = data['inputs'][idx]
phi_true = data['outputs'][idx].reshape(data['metadata'][idx]['phi_shape'])

# BNN prediction
X = torch.tensor([params]).float()
phi_pred_flat, _ = bnn.predict(X)
phi_pred = phi_pred_flat.flatten().cpu().numpy().reshape(phi_true.shape)

# Calculate J
kappa, dy = 5.0, 0.05
phi_surf_true, phi_surf_pred = phi_true[60:120, 0], phi_pred[60:120, 0]
phi_abov_true, phi_abov_pred = phi_true[60:120, 1], phi_pred[60:120, 1]

j_true = kappa * (phi_surf_true - phi_abov_true) / dy
j_pred = kappa * (phi_surf_pred - phi_abov_pred) / dy

print("="*80)
print("CURRENT DENSITY ERROR DEMONSTRATION")
print("="*80)
print(f"\nSample: {data['metadata'][idx]['params']}")
print(f"\n{'Metric':<25} {'Physics':<15} {'BNN':<15} {'Error':<15}")
print("-"*80)

# Phi errors
phi_mae = np.mean(np.abs(phi_true - phi_pred))
phi_rel = phi_mae / np.abs(phi_true).mean() * 100
print(f"{'Phi field (MAE)':<25} {phi_mae:.6f}V     {phi_mae:.6f}V     {phi_rel:.2f}%")

# Gradient errors  
dphi_true_mean = np.mean(phi_surf_true - phi_abov_true)
dphi_pred_mean = np.mean(phi_surf_pred - phi_abov_pred)
dphi_err = abs(dphi_true_mean - dphi_pred_mean)
dphi_rel = dphi_err / abs(dphi_true_mean) * 100
print(f"{'Gradient (mean dphi/dy)':<25} {dphi_true_mean:.6f}V   {dphi_pred_mean:.6f}V   {dphi_rel:.1f}%")

# J errors
j_true_mean = np.mean(j_true)
j_pred_mean = np.mean(j_pred)
j_err = abs(j_true_mean - j_pred_mean)
j_rel = j_err / abs(j_true_mean) * 100
print(f"{'Current density (J)':<25} {j_true_mean:.4e}   {j_pred_mean:.4e}   {j_rel:.1f}%")

print("\n" + "="*80)
print("KEY INSIGHT:")
print("="*80)
print(f"Phi prediction error:  ~{phi_mae:.4f}V")
print(f"Actual gradient:       ~{abs(dphi_true_mean):.4f}V")
print(f"Error/Signal ratio:    ~{phi_mae/abs(dphi_true_mean):.1f}x")
print(f"\n→ Phi error is {phi_mae/abs(dphi_true_mean):.0f}X LARGER than the gradient signal!")
print(f"→ This causes {j_rel:.0f}% error in current density despite {phi_rel:.1f}% phi error")
print("="*80)

print("\nSOLUTION: Retrain BNN to predict [phi, J] directly (not derive J from phi)")
print("See CURRENT_DENSITY_ERROR_ANALYSIS.md for details")
print("="*80)
