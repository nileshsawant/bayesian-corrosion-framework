"""
Dataset Generator for Corrosion BNN Training

Generates a grid of physics simulations over the parameter space:
- NaCl concentration (Molar)
- Temperature (Kelvin)
- pH
- Flow velocity (m/s)

Saves results to a pickle file for later training.
"""

import os
import sys
import numpy as np
import torch
import pickle
from itertools import product

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
from physics_bridge import PhysicsBridge

# Physical Parameter Ranges
PARAM_RANGES = {
    'NaCl': {
        'name': 'NaCl Concentration',
        'unit': 'M (Molar)',
        'min': 0.1,    # Low salinity
        'max': 1.0,    # Saturated brine
        'samples': 4
    },
    'Temp': {
        'name': 'Temperature',
        'unit': 'K',
        'min': 278.0,  # 5°C (cold seawater)
        'max': 313.0,  # 40°C (tropical/warm)
        'samples': 4
    },
    'pH': {
        'name': 'pH',
        'unit': '',
        'min': 6.0,    # Slightly acidic
        'max': 9.0,    # Alkaline
        'samples': 4
    },
    'Flow': {
        'name': 'Flow Velocity',
        'unit': 'm/s',
        'min': 0.1,    # Near-stagnant
        'max': 3.0,    # High flow
        'samples': 4
    }
}

def generate_parameter_grid():
    """
    Generate a grid of parameter combinations.
    
    Returns:
        param_grid: List of parameter combinations
        param_values: Dictionary of parameter arrays
    """
    param_values = {}
    
    print("=" * 60)
    print("PARAMETER RANGES")
    print("=" * 60)
    
    for param_name, config in PARAM_RANGES.items():
        values = np.linspace(config['min'], config['max'], config['samples'])
        param_values[param_name] = values
        
        print(f"{config['name']:20s} ({config['unit']:10s}): ", end='')
        print(f"[{config['min']:.2f} - {config['max']:.2f}]")
        print(f"{'':32s} Values: {values}")
    
    print("=" * 60)
    
    # Generate all combinations
    nacl_vals = param_values['NaCl']
    temp_vals = param_values['Temp']
    ph_vals = param_values['pH']
    flow_vals = param_values['Flow']
    
    param_grid = list(product(nacl_vals, temp_vals, ph_vals, flow_vals))
    
    print(f"Total combinations: {len(param_grid)}")
    print("=" * 60)
    
    return param_grid, param_values

def run_physics_batch(param_grid, repo_path, output_file="training_data.pkl"):
    """
    Run physics simulations for all parameter combinations.
    
    Args:
        param_grid: List of (NaCl, Temp, pH, Flow) tuples
        repo_path: Path to physics repository
        output_file: Output pickle file
    """
    physics = PhysicsBridge(repo_path)
    
    dataset = {
        'inputs': [],
        'outputs': [],
        'metadata': []
    }
    
    total = len(param_grid)
    successful = 0
    failed = 0
    
    print("\nRunning Physics Simulations...")
    print("=" * 60)
    
    for i, params in enumerate(param_grid):
        nacl, temp, ph, flow = params
        
        print(f"\n[{i+1}/{total}] NaCl={nacl:.3f}M, T={temp:.1f}K, pH={ph:.1f}, v={flow:.2f}m/s")
        
        try:
            result = physics.run_sim(list(params))
            
            if result is not None and 'phi' in result:
                # Store input parameters
                input_vec = np.array([nacl, temp, ph, flow])
                
                # Store flattened phi field
                phi_flat = result['phi'].flatten()
                
                # Store current density profile (if available)
                # Concatenate phi and current density as single output vector
                if 'currentDensity' in result:
                    cd_flat = result['currentDensity'].flatten()
                    output_vec = np.concatenate([phi_flat, cd_flat])
                else:
                    output_vec = phi_flat
                
                # Store metadata
                metadata = {
                    'params': {'NaCl': nacl, 'Temp': temp, 'pH': ph, 'Flow': flow},
                    'corrosion_rate': result.get('corrosionRate', None),
                    'phi_shape': result['phi'].shape,
                    'phi_length': len(phi_flat),
                    'current_density_length': len(result.get('currentDensity', [])),
                    'output_length': len(output_vec)
                }
                
                dataset['inputs'].append(input_vec)
                dataset['outputs'].append(output_vec)
                dataset['metadata'].append(metadata)
                
                successful += 1
                cd_info = f", CD length: {metadata['current_density_length']}" if 'currentDensity' in result else ""
                print(f"✓ Success | Phi: {result['phi'].shape}, Corrosion: {metadata['corrosion_rate']:.2e} A/m²{cd_info}")
            else:
                failed += 1
                print(f"✗ Failed: No valid result returned")
                
        except Exception as e:
            failed += 1
            print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {successful} successful, {failed} failed")
    print("=" * 60)
    
    if successful == 0:
        print("ERROR: No successful simulations. Cannot save dataset.")
        return None
    
    # Convert to numpy arrays
    dataset['inputs'] = np.array(dataset['inputs'])
    dataset['outputs'] = np.array(dataset['outputs'])
    
    # Add parameter ranges for reference
    dataset['param_ranges'] = PARAM_RANGES
    
    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "..", output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"Input shape: {dataset['inputs'].shape}")
    print(f"Output shape: {dataset['outputs'].shape}")
    
    return dataset

if __name__ == "__main__":
    # Repository path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                              "../../corrosion-modeling-applications"))
    
    if not os.path.exists(repo_root):
        print(f"ERROR: Physics repository not found at {repo_root}")
        sys.exit(1)
    
    print(f"Physics Repository: {repo_root}\n")
    
    # Generate parameter grid
    param_grid, param_values = generate_parameter_grid()
    
    # Run physics batch
    dataset = run_physics_batch(param_grid, repo_root, output_file="training_data.pkl")
    
    if dataset is not None:
        print("\n✓ Dataset generation complete!")
    else:
        print("\n✗ Dataset generation failed!")
        sys.exit(1)
