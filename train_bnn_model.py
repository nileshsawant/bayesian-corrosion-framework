#!/projects/hpcapps/nsawant/corrosion/env/bin/python3
"""
Script 2: Train BNN Model
=========================
Trains a Bayesian Neural Network on physics simulation data.

Usage:
    python3 train_bnn_model.py --data training_data.pkl --output bnn_model.pt
    
    # Train with more iterations for better accuracy:
    python3 train_bnn_model.py --data training_data_large.pkl --iterations 20000
    
    # Use different architecture:
    python3 train_bnn_model.py --data training_data.pkl --hidden 128,256,128
    
    # Continue training from existing model:
    python3 train_bnn_model.py --data training_data.pkl --model bnn_model.pt --iterations 5000
"""

import argparse
import sys
import os
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_bnn import train_bnn_batch


def main():
    parser = argparse.ArgumentParser(
        description='Train Bayesian Neural Network on corrosion data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (10k iterations):
  python3 train_bnn_model.py --data training_data.pkl
  
  # More iterations for better convergence:
  python3 train_bnn_model.py --data training_data.pkl --iterations 20000
  
  # Larger network for complex data:
  python3 train_bnn_model.py --data training_data_100.pkl --hidden 128,256,128
  
  # Continue training (fine-tuning):
  python3 train_bnn_model.py --data training_data.pkl --model bnn_model.pt --iterations 5000
  
  # Adjust learning rate:
  python3 train_bnn_model.py --data training_data.pkl --lr 0.001 --iterations 15000

Performance Notes:
  - H100 GPU: ~50 it/s, 10k iterations in ~3 minutes
  - Larger datasets benefit from more iterations
  - Monitor loss convergence in output
        """)
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (.pkl file)')
    
    parser.add_argument('--output', type=str, default='bnn_model.pt',
                       help='Output model file (default: bnn_model.pt)')
    
    parser.add_argument('--model', type=str, default=None,
                       help='Existing model to continue training (optional)')
    
    parser.add_argument('--iterations', type=int, default=10000,
                       help='Number of training iterations (default: 10000)')
    
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate (default: 0.005)')
    
    parser.add_argument('--hidden', type=str, default='64,128,64',
                       help='Hidden layer dimensions (comma-separated, default: 64,128,64)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Parse hidden dimensions
    hidden_dims = [int(x) for x in args.hidden.split(',')]
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        print("\nGenerate data first:")
        print("  python3 generate_training_data.py --samples 16")
        sys.exit(1)
    
    print("="*80)
    print("BNN TRAINING")
    print("="*80)
    print(f"Training data:   {args.data}")
    print(f"Output model:    {args.output}")
    print(f"Iterations:      {args.iterations}")
    print(f"Learning rate:   {args.lr}")
    print(f"Hidden layers:   {hidden_dims}")
    print(f"Device:          {args.device}")
    
    if args.model:
        print(f"Continue from:   {args.model}")
    
    print("="*80)
    print()
    
    # Train
    train_bnn_batch(
        data_path=args.data,
        model_path=args.output,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        hidden_dims=hidden_dims,
        device=args.device
    )
    
    print()
    print("="*80)
    print("✓ Training Complete!")
    print("="*80)
    print(f"Model saved to: {args.output}")
    print()
    print("Next step: Run inference")
    print(f"  python3 run_inference.py --model {args.output} --data {args.data}")
    print()
    print("Or generate comparison plots:")
    print(f"  python3 -c \"import sys; sys.path.insert(0, 'src'); from compare_predictions import *; bnn, data = load_model_and_data('{args.output}', '{args.data}'); create_summary_report(bnn, data, num_samples=8)\"")
    print("="*80)


if __name__ == "__main__":
    main()
