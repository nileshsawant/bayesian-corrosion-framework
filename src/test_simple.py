"""
Simple test to debug the hanging issue
"""
import os
import sys
import pickle
import numpy as np

print("0. Imports done (before torch)")
sys.stdout.flush()

import torch
print("0b. Torch imported")
sys.stdout.flush()

sys.path.insert(0, os.path.dirname(__file__))

print("1. Loading BNN...")
sys.stdout.flush()

from bnn_model import BNNWrapper
print("1b. BNNWrapper imported")
sys.stdout.flush()

model_path = os.path.join(os.path.dirname(__file__), "..", "bnn_model.pt")
data_path = os.path.join(os.path.dirname(__file__), "..", "training_data.pkl")

print("2. Loading model from disk...")
sys.stdout.flush()

bnn = BNNWrapper.load(model_path)
print("2b. Model loaded")
sys.stdout.flush()

print("3. Loading dataset...")
sys.stdout.flush()

with open(data_path, 'rb') as f:
    dataset = pickle.load(f)
print("3b. Dataset loaded")
sys.stdout.flush()

print("4. Preparing test input...")
sys.stdout.flush()

params = dataset['inputs'][0]
X = torch.tensor([params]).float()
print("4b. Tensor created")
sys.stdout.flush()

print("5. Running prediction (this may take a moment)...")
sys.stdout.flush()

with torch.no_grad():
    pred_mean, pred_std = bnn.predict(X, num_samples=10)  # Use fewer samples
    
print("5b. Prediction complete")
sys.stdout.flush()

print("6. Success!")
print(f"   Prediction shape: {pred_mean.shape}")
print(f"   Mean value: {pred_mean.mean():.6f}")
print(f"   Std value: {pred_std.mean():.6f}")
sys.stdout.flush()
