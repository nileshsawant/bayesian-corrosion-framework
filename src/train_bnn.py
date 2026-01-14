"""
Train BNN on Pre-Generated Dataset

Loads the dataset from generate_dataset.py and trains the BNN
on all samples at once (supervised learning mode).
"""

import os
import sys
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bnn_model import BNNWrapper

def load_dataset(dataset_path):
    """Load the training dataset."""
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print("=" * 60)
    print("DATASET LOADED")
    print("=" * 60)
    print(f"Number of samples: {len(dataset['inputs'])}")
    print(f"Input shape: {dataset['inputs'].shape}")
    print(f"Output shape: {dataset['outputs'].shape}")
    print("=" * 60)
    
    return dataset

def train_bnn_batch(dataset=None, data_path=None, model_path="bnn_model.pt", 
                    num_iterations=5000, device='auto', learning_rate=0.003, 
                    hidden_dims=[128, 256, 128]):
    """
    Train BNN on full dataset.
    
    Args:
        dataset: Dictionary with 'inputs' and 'outputs' (or None to load from data_path)
        data_path: Path to dataset pickle file (if dataset not provided)
        model_path: Path to save trained model (can be relative or absolute)
        num_iterations: Training iterations (default 5000 for better convergence)
        device: 'cuda', 'cpu', or 'auto'
        learning_rate: Learning rate for Adam optimizer (default 0.005)
        hidden_dims: List of hidden layer dimensions (default [64, 128, 64])
    """
    # Load dataset if path provided
    if dataset is None and data_path is not None:
        dataset = load_dataset(data_path)
    elif dataset is None:
        raise ValueError("Must provide either 'dataset' or 'data_path'")
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"\nTraining Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Convert to tensors and move to device
    X = torch.tensor(dataset['inputs']).float().to(device)
    Y = torch.tensor(dataset['outputs']).float().to(device)
    
    n_samples = X.shape[0]
    output_dim = Y.shape[1]
    
    print(f"\nTraining on {n_samples} samples")
    print(f"Input dimensions: {X.shape[1]}")
    print(f"Output dimensions: {output_dim}")
    print(f"Hidden layers: {hidden_dims}")
    print(f"Learning rate: {learning_rate}")
    
    # Initialize BNN with GPU support and specified architecture
    bnn = BNNWrapper(input_dim=4, output_dim=output_dim, hidden_dims=hidden_dims, 
                     device=device)
    
    # Set learning rate (create new optimizer with specified LR)
    from pyro.optim import Adam
    from pyro.infer import SVI, Trace_ELBO
    optimizer = Adam({"lr": learning_rate})
    bnn.bnn.svi = SVI(bnn.bnn.model, bnn.bnn.guide, optimizer, loss=Trace_ELBO())
    
    # Normalize outputs ONCE before training (not in the loop!)
    Y_norm = bnn.bnn.normalize_output(Y)
    
    # Train
    print(f"\nStarting training for {num_iterations} iterations...")
    print("=" * 60)
    
    import time
    start_time = time.time()
    losses = []
    
    for i in range(num_iterations):
        # Batch training with pre-normalized Y
        loss = bnn.bnn.svi.step(X, Y_norm)
        losses.append(loss)
        
        if (i + 1) % 500 == 0:
            recent_loss = sum(losses[-500:]) / 500 if len(losses) >= 500 else sum(losses) / len(losses)
            elapsed = time.time() - start_time
            iter_per_sec = (i + 1) / elapsed
            eta = (num_iterations - i - 1) / iter_per_sec
            
            # Show GPU memory usage if on CUDA
            gpu_info = ""
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(0) / 1e9
                mem_reserved = torch.cuda.memory_reserved(0) / 1e9
                gpu_info = f" | GPU Mem: {mem_allocated:.1f}/{mem_reserved:.1f}GB"
            
            print(f"[Step {i+1}/{num_iterations}] Loss: {recent_loss:.2f} | {iter_per_sec:.1f} it/s | ETA: {eta/60:.1f}m{gpu_info}")
    
    total_time = time.time() - start_time
    
    print("=" * 60)
    print(f"Training Complete!")
    print(f"Total Time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Average Speed: {num_iterations/total_time:.1f} iterations/second")
    print(f"Initial Loss: {losses[0]:.2f}")
    print(f"Final Loss: {losses[-1]:.2f}")
    print(f"Loss Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    # Report learned noise parameters
    try:
        import pyro
        sigma_phi_loc = pyro.param('sigma_phi_loc').item()
        sigma_phi_scale = pyro.param('sigma_phi_scale').item()
        sigma_j_loc = pyro.param('sigma_j_loc').item()
        sigma_j_scale = pyro.param('sigma_j_scale').item()
        
        # Convert to actual sigma values (median of LogNormal)
        sigma_phi_median = np.exp(sigma_phi_loc)
        sigma_j_median = np.exp(sigma_j_loc)
        
        print(f"\nLearned Noise Parameters:")
        print(f"  Phi noise (σ_φ):  {sigma_phi_median:.4f} (loc={sigma_phi_loc:.4f}, scale={sigma_phi_scale:.4f})")
        print(f"  J noise (σ_J):    {sigma_j_median:.4f} (loc={sigma_j_loc:.4f}, scale={sigma_j_scale:.4f})")
        print(f"  Ratio (σ_J/σ_φ):  {sigma_j_median/sigma_phi_median:.2f}x")
    except:
        pass
    
    print("=" * 60)
    
    # Save model using wrapper method
    model_full_path = os.path.join(os.path.dirname(__file__), "..", model_path)
    bnn.save(model_full_path)
    
    print(f"\nModel saved to: {model_full_path}")
    
    return bnn

def test_inference(bnn, dataset):
    """
    Test inference on a few samples.
    """
    # Get device from BNN
    device = bnn.bnn.device
    
    X = torch.tensor(dataset['inputs'][:5]).float().to(device)
    Y = torch.tensor(dataset['outputs'][:5]).float().to(device)
    
    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)
    
    with torch.no_grad():
        pred_mean, pred_std = bnn.predict(X, num_samples=50)
        
        for i in range(len(X)):
            params = dataset['metadata'][i]['params']
            
            # Compute metrics
            y_true = Y[i]
            y_pred = pred_mean[i]
            
            mse = ((y_true - y_pred)**2).mean().item()
            mae = (y_true - y_pred).abs().mean().item()
            
            print(f"\nSample {i+1}:")
            print(f"  Params: NaCl={params['NaCl']:.2f}M, T={params['Temp']:.1f}K, "
                  f"pH={params['pH']:.1f}, v={params['Flow']:.2f}m/s")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  Mean uncertainty: {pred_std[i].mean().item():.6f}")
    
    print("=" * 60)

if __name__ == "__main__":
    # Paths
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "training_data.pkl")
    model_path = "bnn_model.pt"
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run generate_dataset.py first!")
        sys.exit(1)
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Train BNN with more iterations for better convergence
    bnn = train_bnn_batch(dataset, model_path=model_path, num_iterations=10000)
    
    # Test inference
    test_inference(bnn, dataset)
    
    print("\n✓ Training and testing complete!")
