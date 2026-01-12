#!/projects/hpcapps/nsawant/corrosion/env/bin/python3
"""
Script 1: Generate Training Data
================================
Generates training data by running physics simulations across parameter space.
Outputs concatenated [phi, J] vectors for each sample.

Usage:
    python3 generate_training_data.py --samples 16 --output training_data.pkl
    
    # For more data (e.g., 100 samples):
    python3 generate_training_data.py --samples 100 --output training_data_large.pkl
    
    # For specific materials:
    python3 generate_training_data.py --samples 50 --materials HY80 HY100 --output hy_steels.pkl
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from physics_bridge import generate_training_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data from physics simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 16 samples (default, quick test):
  python3 generate_training_data.py
  
  # Generate 100 samples for better accuracy:
  python3 generate_training_data.py --samples 100 --output training_data_100.pkl
  
  # Generate data for specific materials:
  python3 generate_training_data.py --samples 50 --materials HY80 HY100
  
  # Custom parameter ranges:
  python3 generate_training_data.py --samples 32 --nacl 0.1,0.5,1.0 --temp 278,295,313
        """)
    
    parser.add_argument('--samples', type=int, default=16,
                       help='Number of samples to generate (default: 16)')
    
    parser.add_argument('--output', type=str, default='training_data.pkl',
                       help='Output file path (default: training_data.pkl)')
    
    parser.add_argument('--materials', type=str, nargs='+', 
                       default=['HY80'],
                       help='Materials to simulate (default: HY80). Options: HY80, HY100, SS316, I625, CuNi, Ti')
    
    parser.add_argument('--nacl', type=str, default='0.1,1.0',
                       help='NaCl concentrations in M (comma-separated, default: 0.1,1.0)')
    
    parser.add_argument('--temp', type=str, default='278,313',
                       help='Temperatures in K (comma-separated, default: 278,313)')
    
    parser.add_argument('--ph', type=str, default='6.0,9.0',
                       help='pH values (comma-separated, default: 6.0,9.0)')
    
    parser.add_argument('--flow', type=str, default='0.1,3.0',
                       help='Flow velocities in m/s (comma-separated, default: 0.1,3.0)')
    
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    # Parse parameter ranges
    nacl_values = [float(x) for x in args.nacl.split(',')]
    temp_values = [float(x) for x in args.temp.split(',')]
    ph_values = [float(x) for x in args.ph.split(',')]
    flow_values = [float(x) for x in args.flow.split(',')]
    
    print("="*80)
    print("TRAINING DATA GENERATION")
    print("="*80)
    print(f"Target samples:  {args.samples}")
    print(f"Output file:     {args.output}")
    print(f"Materials:       {', '.join(args.materials)}")
    print(f"NaCl (M):        {nacl_values}")
    print(f"Temperature (K): {temp_values}")
    print(f"pH:              {ph_values}")
    print(f"Flow (m/s):      {flow_values}")
    print(f"Max workers:     {args.max_workers}")
    print("="*80)
    print()
    
    # Generate parameter combinations
    import itertools
    param_combinations = []
    
    for material in args.materials:
        for nacl, temp, ph, flow in itertools.product(
            nacl_values, temp_values, ph_values, flow_values
        ):
            param_combinations.append({
                'material': material,
                'NaCl': nacl,
                'Temp': temp,
                'pH': ph,
                'Flow': flow
            })
    
    # Sample if we have more combinations than requested
    if len(param_combinations) > args.samples:
        import numpy as np
        np.random.seed(42)
        indices = np.random.choice(len(param_combinations), args.samples, replace=False)
        param_combinations = [param_combinations[i] for i in sorted(indices)]
    
    print(f"Generating {len(param_combinations)} parameter combinations...")
    print()
    
    # Generate dataset
    dataset = generate_training_dataset(
        param_combinations,
        max_workers=args.max_workers
    )
    
    # Save
    import pickle
    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)
    
    print()
    print("="*80)
    print("✓ Dataset Generation Complete!")
    print("="*80)
    print(f"Saved to: {args.output}")
    print(f"Samples:  {len(dataset['inputs'])}")
    print(f"Inputs:   {dataset['inputs'].shape}")
    print(f"Outputs:  {dataset['outputs'].shape}")
    print()
    print("Next step: Train BNN")
    print(f"  python3 train_bnn_model.py --data {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
