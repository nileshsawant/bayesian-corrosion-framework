
import torch
import pyro
from bnn_model import BNNWrapper
import numpy as np

def test_convergence():
    input_dim = 4
    output_dim = 100 # Smaller for debug
    
    print("Initializing BNN...")
    bnn = BNNWrapper(input_dim=input_dim, output_dim=output_dim)
    
    # Dummy Data: Input [1.0, ...] -> Output [0.5, 0.5, ...]
    x_train = torch.randn(1, input_dim)
    y_train = torch.ones(1, output_dim) * 0.5
    
    print("Initial Prediction (Before Training):")
    mean, std = bnn.predict(x_train, num_samples=10)
    print(f"Mean (First 5): {mean[0, :5]}")
    print(f"MSE: {torch.nn.functional.mse_loss(mean, y_train).item()}")
    
    print("\nTraining...")
    # Train heavily to force overfitting
    losses = []
    pyro.clear_param_store()
    
    for i in range(500):
        loss = bnn.svi.step(x_train, y_train)
        losses.append(loss)
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss:.4f}")
            
    print(f"Final Loss: {losses[-1]:.4f}")
    
    print("\nFinal Prediction (After Training):")
    mean, std = bnn.predict(x_train, num_samples=50)
    print(f"Mean (First 5): {mean[0, :5]}")
    mse = torch.nn.functional.mse_loss(mean, y_train).item()
    print(f"MSE: {mse}")
    
    if mse < 0.05:
        print("SUCCESS: Model converged.")
    else:
        print("FAILURE: Model did not converge.")

if __name__ == "__main__":
    test_convergence()
