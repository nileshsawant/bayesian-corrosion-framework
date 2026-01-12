#!/usr/bin/env python3
"""
Quick test to verify GPU utilization and training speed after optimization.
"""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))

print("="*80)
print("GPU & TRAINING SPEED TEST")
print("="*80)

# Check GPU availability
print("\n1. GPU Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("   WARNING: CUDA not available! Training will be SLOW on CPU.")

# Load dataset
print("\n2. Loading Dataset:")
import pickle
data_path = "/projects/hpcapps/nsawant/corrosion/bayesian-corrosion-framework/training_data.pkl"
with open(data_path, 'rb') as f:
    dataset = pickle.load(f)

print(f"   Samples: {len(dataset['inputs'])}")
print(f"   Input dim: {dataset['inputs'].shape[1]}")
print(f"   Output dim: {dataset['outputs'].shape[1]}")

# Quick training test (100 iterations)
print("\n3. Training Speed Test (100 iterations):")
print("   This will help verify GPU is being used efficiently...")

from train_bnn import train_bnn_batch

start = time.time()
bnn = train_bnn_batch(
    dataset=dataset,
    model_path="/tmp/test_bnn.pt",
    num_iterations=100,
    device='auto',
    learning_rate=0.005,
    hidden_dims=[64, 128, 64]
)
elapsed = time.time() - start

print(f"\n4. Results:")
print(f"   100 iterations completed in {elapsed:.1f} seconds")
print(f"   Speed: {100/elapsed:.1f} iterations/second")
print(f"   Estimated time for 10,000 iterations: {(elapsed * 100)/60:.1f} minutes")

if torch.cuda.is_available():
    mem_allocated = torch.cuda.memory_allocated(0) / 1e9
    mem_cached = torch.cuda.memory_reserved(0) / 1e9
    print(f"   GPU Memory Used: {mem_allocated:.2f} GB (cached: {mem_cached:.2f} GB)")
    
    if 100/elapsed < 5:
        print("\n   ⚠ WARNING: Training is slower than expected!")
        print("   Expected: >20 it/s on H100 GPU")
        print("   Possible issues:")
        print("     - Data not on GPU")
        print("     - Network architecture too large")
        print("     - CPU bottleneck in data loading")
    elif 100/elapsed > 20:
        print("\n   ✓ Training speed is EXCELLENT!")
        print(f"   Full training (10k iterations) should take ~{(10000/elapsed)/60:.0f} minutes")
    else:
        print("\n   ✓ Training speed is good")

print("\n" + "="*80)
