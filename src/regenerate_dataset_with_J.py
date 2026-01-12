#!/usr/bin/env python3
"""
Regenerate training dataset with current density profiles included.
Uses the same 16 parameter combinations as before but includes J profiles.
"""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics_bridge import PhysicsBridge

# Load existing dataset to get parameters
old_data_path = "/projects/hpcapps/nsawant/corrosion/bayesian-corrosion-framework/training_data.pkl"
print("Loading existing dataset to extract parameters...")
with open(old_data_path, 'rb') as f:
    old_data = pickle.load(f)

param_combinations = []
for meta in old_data['metadata']:
    p = meta['params']
    param_combinations.append([p['NaCl'], p['Temp'], p['pH'], p['Flow']])

print(f"Found {len(param_combinations)} parameter combinations")
print("\nRegenerating dataset with current density profiles...")
print("="*80)

# Initialize physics bridge
repo_path = "/projects/hpcapps/nsawant/corrosion/corrosion-modeling-applications"
physics = PhysicsBridge(repo_path)

# New dataset structure
dataset = {
    'inputs': [],
    'outputs': [],
    'metadata': [],
    'param_ranges': old_data.get('param_ranges', {})
}

successful = 0
failed = 0

for i, params in enumerate(param_combinations):
    nacl, temp, ph, flow = params
    print(f"\n[{i+1}/{len(param_combinations)}] NaCl={nacl:.2f}M, T={temp:.1f}K, pH={ph:.1f}, v={flow:.2f}m/s")
    
    try:
        result = physics.run_sim(params)
        
        if result is not None and 'phi' in result and 'currentDensity' in result:
            # Store input
            input_vec = np.array([nacl, temp, ph, flow])
            
            # Store concatenated output: [phi_field, J_profile]
            phi_flat = result['phi'].flatten()
            j_flat = result['currentDensity'].flatten()
            output_vec = np.concatenate([phi_flat, j_flat])
            
            # Store metadata
            metadata = {
                'params': {'NaCl': nacl, 'Temp': temp, 'pH': ph, 'Flow': flow},
                'corrosion_rate': result.get('corrosionRate', None),
                'phi_shape': result['phi'].shape,
                'phi_length': len(phi_flat),
                'current_density_length': len(j_flat),
                'output_length': len(output_vec)
            }
            
            dataset['inputs'].append(input_vec)
            dataset['outputs'].append(output_vec)
            dataset['metadata'].append(metadata)
            
            successful += 1
            print(f"  ✓ Success | Phi: {result['phi'].shape}, J: {len(j_flat)}, "
                  f"Corr: {metadata['corrosion_rate']:.2e} A/m²")
        else:
            failed += 1
            has_phi = 'phi' in result if result else False
            has_j = 'currentDensity' in result if result else False
            print(f"  ✗ Failed: phi={has_phi}, J={has_j}")
            
    except Exception as e:
        failed += 1
        print(f"  ✗ Failed: {e}")

print("\n" + "="*80)
print(f"SUMMARY: {successful} successful, {failed} failed")
print("="*80)

if successful == 0:
    print("ERROR: No successful simulations!")
    sys.exit(1)

# Convert to numpy
dataset['inputs'] = np.array(dataset['inputs'])
dataset['outputs'] = np.array(dataset['outputs'])

print(f"\nDataset shapes:")
print(f"  Inputs:  {dataset['inputs'].shape}")
print(f"  Outputs: {dataset['outputs'].shape}")
print(f"  Output breakdown: {dataset['metadata'][0]['phi_length']} (phi) + "
      f"{dataset['metadata'][0]['current_density_length']} (J) = "
      f"{dataset['metadata'][0]['output_length']}")

# Backup old dataset
backup_path = old_data_path.replace('.pkl', '_without_J.pkl')
print(f"\nBacking up old dataset to: {backup_path}")
os.rename(old_data_path, backup_path)

# Save new dataset
print(f"Saving new dataset to: {old_data_path}")
with open(old_data_path, 'wb') as f:
    pickle.dump(dataset, f)

print("\n✓✓✓ Dataset regeneration complete!")
print("\nNext steps:")
print("  1. Retrain BNN with: python3 train_bnn.py")
print("  2. Run comparison: python3 compare_predictions.py")
