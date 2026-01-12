"""
Quick Test Dataset Generator

Generates a small dataset (2x2x2x2 = 16 samples) for testing the pipeline.
"""

import os
import sys

# Modify the parameter ranges to use only 2 samples per parameter
sys.path.insert(0, os.path.dirname(__file__))

# Import and modify the parameter ranges
import generate_dataset as gd

# Override with smaller grid for testing
gd.PARAM_RANGES['NaCl']['samples'] = 2
gd.PARAM_RANGES['Temp']['samples'] = 2
gd.PARAM_RANGES['pH']['samples'] = 2
gd.PARAM_RANGES['Flow']['samples'] = 2

print("=" * 60)
print("QUICK TEST MODE")
print("=" * 60)
print("Using 2 values per parameter = 16 total simulations")
print("This should complete in ~30-60 minutes")
print("=" * 60)

if __name__ == "__main__":
    # Repository path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                              "../../corrosion-modeling-applications"))
    
    if not os.path.exists(repo_root):
        print(f"ERROR: Physics repository not found at {repo_root}")
        sys.exit(1)
    
    print(f"Physics Repository: {repo_root}\n")
    
    # Generate parameter grid
    param_grid, param_values = gd.generate_parameter_grid()
    
    # Run physics batch
    dataset = gd.run_physics_batch(param_grid, repo_root, output_file="training_data.pkl")
    
    if dataset is not None:
        print("\n✓ Test dataset generation complete!")
    else:
        print("\n✗ Test dataset generation failed!")
        sys.exit(1)
