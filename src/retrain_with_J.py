#!/usr/bin/env python3
"""
Quick retraining script for BNN with [phi, J] outputs.
Automatically detects output dimension from dataset.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from train_bnn import train_bnn_batch

if __name__ == "__main__":
    print("="*80)
    print("RETRAINING BNN WITH [PHI, J] OUTPUTS")
    print("="*80)
    
    data_path = "/projects/hpcapps/nsawant/corrosion/bayesian-corrosion-framework/training_data.pkl"
    model_path = "/projects/hpcapps/nsawant/corrosion/bayesian-corrosion-framework/bnn_model.pt"
    
    # Train with more iterations for better convergence
    print("\nStarting training with 10,000 iterations...")
    print("(This may take 10-15 minutes with GPU initialization)")
    print("-"*80)
    
    train_bnn_batch(
        data_path=data_path,
        model_save_path=model_path,
        num_iterations=10000,
        learning_rate=0.005,
        hidden_dims=[64, 128, 64]
    )
    
    print("\n" + "="*80)
    print("✓✓✓ RETRAINING COMPLETE!")
    print("="*80)
    print("\nNext step: Run comparison to see improved J predictions")
    print("  python3 compare_predictions.py")
    print("="*80)
