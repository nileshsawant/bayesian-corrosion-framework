#!/projects/hpcapps/nsawant/corrosion/env/bin/python3
"""
Script 3: Run BNN Inference
============================
Runs inference with trained BNN model on new parameter combinations.

Usage:
    python3 run_inference.py --model bnn_model.pt --params params.txt
    
    # Single prediction from command line:
    python3 run_inference.py --model bnn_model.pt --nacl 0.5 --temp 295 --ph 7.5 --flow 1.0
    
    # Batch predictions with uncertainty:
    python3 run_inference.py --model bnn_model.pt --params params.txt --samples 500
    
    # Compare with training data:
    python3 run_inference.py --model bnn_model.pt --data training_data.pkl --compare
"""

import argparse
import sys
import os
import numpy as np
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bnn_model import BNNWrapper


def print_prediction(params, phi_mean, phi_std, j_mean, j_std, phi_shape=(121, 21)):
    """Pretty print prediction results"""
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(f"Input Parameters:")
    print(f"  NaCl:        {params['NaCl']:.3f} M")
    print(f"  Temperature: {params['Temp']:.1f} K ({params['Temp']-273.15:.1f}°C)")
    print(f"  pH:          {params['pH']:.2f}")
    print(f"  Flow:        {params['Flow']:.3f} m/s")
    print()
    
    # Phi field statistics
    print(f"Potential Field (phi):")
    print(f"  Shape:       {phi_shape[0]} x {phi_shape[1]} grid")
    print(f"  Mean ± Std:  {phi_mean.mean():.4f} ± {phi_std.mean():.4f} V")
    print(f"  Range:       [{phi_mean.min():.4f}, {phi_mean.max():.4f}] V")
    print(f"  Uncertainty: {phi_std.mean():.4f} V (avg), {phi_std.max():.4f} V (max)")
    print()
    
    # Current density profile statistics
    corr_rate_mean = j_mean.mean()
    corr_rate_std = np.sqrt(np.mean(j_std**2))  # Propagate uncertainty
    
    print(f"Current Density Profile (J):")
    print(f"  Length:          {len(j_mean)} points along anode")
    print(f"  Corrosion Rate:  {corr_rate_mean:.4e} ± {corr_rate_std:.4e} A/m²")
    print(f"  Peak Current:    {j_mean.min():.4e} A/m² (most negative)")
    print(f"  Avg Uncertainty: {j_std.mean():.4e} A/m²")
    print()
    
    # Uncertainty as percentage
    phi_uncertainty_pct = 100 * phi_std.mean() / np.abs(phi_mean).mean()
    j_uncertainty_pct = 100 * corr_rate_std / np.abs(corr_rate_mean)
    
    print(f"Prediction Confidence:")
    print(f"  Phi uncertainty: {phi_uncertainty_pct:.2f}%")
    print(f"  J uncertainty:   {j_uncertainty_pct:.2f}%")
    print("="*80)


def run_single_prediction(model_path, params, num_samples=100):
    """Run prediction for single parameter set"""
    # Load model
    print(f"Loading model from: {model_path}")
    bnn = BNNWrapper.load(model_path)
    
    # Prepare input
    param_array = np.array([[
        params['NaCl'],
        params['Temp'],
        params['pH'],
        params['Flow']
    ]])
    
    X = torch.tensor(param_array).float()
    
    # Run prediction
    print(f"Running inference with {num_samples} posterior samples...")
    with torch.no_grad():
        pred_mean, pred_std = bnn.predict(X, num_samples=num_samples)
    
    pred_mean = pred_mean[0].cpu().numpy()
    pred_std = pred_std[0].cpu().numpy()
    
    # Get metadata for parsing (assume standard shapes)
    phi_len = 121 * 21  # Standard grid size
    j_len = 60  # Standard anode length
    
    phi_mean = pred_mean[:phi_len]
    phi_std = pred_std[:phi_len]
    j_mean = pred_mean[phi_len:phi_len+j_len]
    j_std = pred_std[phi_len:phi_len+j_len]
    
    print_prediction(params, phi_mean, phi_std, j_mean, j_std)
    
    return pred_mean, pred_std


def run_batch_predictions(model_path, param_file, num_samples=100):
    """Run predictions for batch of parameters from file"""
    # Load model
    print(f"Loading model from: {model_path}")
    bnn = BNNWrapper.load(model_path)
    
    # Read parameters
    print(f"Reading parameters from: {param_file}")
    params_list = []
    
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse: NaCl,Temp,pH,Flow
            values = [float(x) for x in line.split(',')]
            if len(values) != 4:
                print(f"WARNING: Skipping invalid line: {line}")
                continue
            
            params_list.append({
                'NaCl': values[0],
                'Temp': values[1],
                'pH': values[2],
                'Flow': values[3]
            })
    
    print(f"Loaded {len(params_list)} parameter sets")
    print()
    
    # Run predictions
    param_array = np.array([[p['NaCl'], p['Temp'], p['pH'], p['Flow']] 
                            for p in params_list])
    X = torch.tensor(param_array).float()
    
    print(f"Running batch inference with {num_samples} posterior samples...")
    with torch.no_grad():
        pred_mean, pred_std = bnn.predict(X, num_samples=num_samples)
    
    pred_mean = pred_mean.cpu().numpy()
    pred_std = pred_std.cpu().numpy()
    
    # Print results for each
    phi_len = 121 * 21
    j_len = 60
    
    for i, params in enumerate(params_list):
        phi_mean = pred_mean[i, :phi_len]
        phi_std = pred_std[i, :phi_len]
        j_mean = pred_mean[i, phi_len:phi_len+j_len]
        j_std = pred_std[i, phi_len:phi_len+j_len]
        
        print(f"\n{'='*80}")
        print(f"PREDICTION {i+1}/{len(params_list)}")
        print_prediction(params, phi_mean, phi_std, j_mean, j_std)
    
    return pred_mean, pred_std


def compare_with_training_data(model_path, data_path):
    """Compare predictions with training data"""
    import pickle
    from compare_predictions import load_model_and_data, create_summary_report
    
    print(f"Loading model and data...")
    bnn, dataset = load_model_and_data(model_path, data_path)
    
    print()
    create_summary_report(bnn, dataset, num_samples=min(16, len(dataset['inputs'])))


def main():
    parser = argparse.ArgumentParser(
        description='Run BNN inference for corrosion predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction from command line:
  python3 run_inference.py --model bnn_model.pt --nacl 0.5 --temp 295 --ph 7.5 --flow 1.5
  
  # Batch predictions from file:
  python3 run_inference.py --model bnn_model.pt --params new_conditions.txt
  
  # Higher uncertainty estimates (more posterior samples):
  python3 run_inference.py --model bnn_model.pt --nacl 0.3 --temp 300 --ph 8.0 --flow 2.0 --samples 500
  
  # Compare with training data:
  python3 run_inference.py --model bnn_model.pt --data training_data.pkl --compare

Parameter file format (CSV):
  # NaCl(M), Temp(K), pH, Flow(m/s)
  0.1, 278, 6.0, 0.1
  0.5, 295, 7.5, 1.5
  1.0, 313, 9.0, 3.0
        """)
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    
    parser.add_argument('--params', type=str, default=None,
                       help='Path to parameter file (CSV format)')
    
    parser.add_argument('--data', type=str, default=None,
                       help='Training data for comparison (optional)')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare predictions with training data')
    
    # Single prediction parameters
    parser.add_argument('--nacl', type=float, default=None,
                       help='NaCl concentration (M)')
    parser.add_argument('--temp', type=float, default=None,
                       help='Temperature (K)')
    parser.add_argument('--ph', type=float, default=None,
                       help='pH value')
    parser.add_argument('--flow', type=float, default=None,
                       help='Flow velocity (m/s)')
    
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of posterior samples for uncertainty (default: 100)')
    
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("\nTrain model first:")
        print("  python3 train_bnn_model.py --data training_data.pkl")
        sys.exit(1)
    
    print("="*80)
    print("BNN INFERENCE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Posterior samples: {args.samples}")
    print("="*80)
    
    # Mode 1: Compare with training data
    if args.compare and args.data:
        compare_with_training_data(args.model, args.data)
    
    # Mode 2: Batch predictions from file
    elif args.params:
        run_batch_predictions(args.model, args.params, args.samples)
    
    # Mode 3: Single prediction from command line
    elif all([args.nacl is not None, args.temp is not None, 
              args.ph is not None, args.flow is not None]):
        params = {
            'NaCl': args.nacl,
            'Temp': args.temp,
            'pH': args.ph,
            'Flow': args.flow
        }
        run_single_prediction(args.model, params, args.samples)
    
    else:
        print("ERROR: Must specify either:")
        print("  1. --compare --data <training_data.pkl>")
        print("  2. --params <param_file.txt>")
        print("  3. --nacl --temp --ph --flow (all four required)")
        print("\nSee --help for examples")
        sys.exit(1)


if __name__ == "__main__":
    main()
