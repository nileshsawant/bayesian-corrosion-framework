import numpy as np
import os
import sys
import torch
import pyro

# Code removed: Placeholder classes CorrosionBNN and PhysicsSimulator were deleted 
# as they are now replaced by imports.

from bnn_model import BNNWrapper
from physics_bridge import PhysicsBridge
try:
    from plotting_utils import plot_simulation_results
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

class Orchestrator:
    MODEL_VERSION = "v5"  # Increment when architecture changes
    
    def __init__(self, uncertainty_threshold=0.1, repo_path="dependencies/corrosion-modeling-applications", model_path="bnn_model.pt"):
        self.bnn = BNNWrapper(input_dim=4)
        self.physics = PhysicsBridge(repo_path)
        self.threshold = uncertainty_threshold
        self.model_path = model_path
        self.version_path = model_path + ".version"
        
        # Normalization Constants (Approximate)
        # [NaCl, Temp, pH, Flow]
        self.input_mean = torch.tensor([0.3, 298.0, 7.0, 1.0])
        self.input_std = torch.tensor([0.3, 20.0, 3.0, 1.0])
        
        # Load existing model if available and compatible
        if os.path.exists(self.model_path):
            # Check version compatibility
            saved_version = None
            if os.path.exists(self.version_path):
                try:
                    with open(self.version_path, 'r') as f:
                        saved_version = f.read().strip()
                except:
                    pass
            
            if saved_version != self.MODEL_VERSION:
                print(f"Model architecture version mismatch (saved: {saved_version}, current: {self.MODEL_VERSION})")
                print(f"Deleting incompatible model and starting fresh...")
                os.remove(self.model_path)
                if os.path.exists(self.version_path):
                    os.remove(self.version_path)
            else:
                print(f"Loading existing BNN model from {self.model_path}")
                try:
                    state_dict = torch.load(self.model_path, weights_only=False)
                    pyro.get_param_store().set_state(state_dict)
                    print("Model loaded successfully.")
                except Exception as e:
                    print(f"Warning: Could not load model: {e}")
                    print("Starting with fresh model (random weights).")
                    os.remove(self.model_path)
                    if os.path.exists(self.version_path):
                        os.remove(self.version_path)

    def save_model(self):
        print(f"Saving BNN model to {self.model_path}...")
        pyro.get_param_store().save(self.model_path)
        # Save version info
        with open(self.version_path, 'w') as f:
            f.write(self.MODEL_VERSION)

    def normalize_input(self, env_tensor):
        if env_tensor.dim() == 1:
            return (env_tensor - self.input_mean) / self.input_std
        return (env_tensor - self.input_mean.unsqueeze(0)) / self.input_std.unsqueeze(0)

    def get_corrosion_prediction(self, env_vector):
        # env_vector should be a Tensor for BNN
        if not isinstance(env_vector, torch.Tensor):
            env_tensor = torch.tensor(env_vector).float()
        else:
            env_tensor = env_vector.float()
            
        # Normalize Input for BNN
        norm_env_tensor = self.normalize_input(env_tensor)
            
        if norm_env_tensor.dim() == 1:
            norm_env_tensor = norm_env_tensor.unsqueeze(0)
            
        # 1. Fast Path
        pred_mean, pred_std = self.bnn.predict(norm_env_tensor)
        
        # Calculate uncertainty metric (Mean Std Dev across the field)
        uncertainty = pred_std.mean().item()
        
        # 2. Decision Gate
        print(f"Field Uncertainty: {uncertainty:.4f}")
        
        if uncertainty < self.threshold:
            print("High Confidence: Returning BNN Prediction")
            return pred_mean, pred_std, None
        
        # 3. Slow Path
        print("Low Confidence: Triggering Physics Engine")
        # Convert tensor back to list for physics bridge (Use unnormalized Original)
        if env_tensor.dim() > 1:
             env_list = env_tensor.squeeze().tolist()
        else:
             env_list = env_tensor.tolist()

        result_dict = self.physics.run_sim(env_list)
        
        # 4. Active Learning
        if result_dict is not None and 'phi' in result_dict:
            print("Training BNN with new Ground Truth...")
            
            # Extract and flatten the 2D field
            # Physics returns phi as (121, 21). Flatten with C-order (PyTorch default)
            phi_field = result_dict['phi']
            phi_flat = torch.tensor(phi_field).float().flatten().unsqueeze(0)
            
            # Update Model (Train on Normalized Input)
            loss = self.bnn.train_step(norm_env_tensor, phi_flat)
            print(f"Model updated. Loss: {loss:.4f}")
            
            self.save_model()
            
            # Return BNN prediction AND Physics Truth for comparison
            # First arg is BNN prediction, Third arg is Physics Truth Dictionary
            return pred_mean, pred_std, result_dict
        
        return None, None, None

if __name__ == "__main__":
    # Example Usage
    # Point to the location where the dependency was cloned
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../corrosion-modeling-applications"))
    
    if not os.path.exists(repo_root):
        print(f"CRITICAL: Dependency not found at {repo_root}")
        sys.exit(1)
    
    print(f"Initializing Orchestrator with repo: {repo_root}")
    system = Orchestrator(uncertainty_threshold=0.001, repo_path=repo_root)
    
    # Test conditions [NaCl, Temp, pH, Flow]
    # Standard Seawater: 0.6M NaCl, 298K (25C), pH 8.2, 1 m/s
    test_env = torch.tensor([0.6, 298.0, 8.2, 1.0])
    
    print("\n--- Starting Active Learning Query ---")
    mean_field, std_field, physics_data = system.get_corrosion_prediction(test_env)
    
    if mean_field is not None:
        print(f"\nResult Obtained.")
        
        output_dir = os.path.join(os.path.dirname(__file__), "../results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Helper to make dummy coordinates for BNN data
        def make_bnn_sim_data(tensor_field):
             phi_data = tensor_field.detach().cpu().numpy() if isinstance(tensor_field, torch.Tensor) else tensor_field
             
             # Reshape if flattened (1, 2541) -> (121, 21)
             # Physics engine produces phi as (NX=121, NY=21) where rows=X, cols=Y
             # After flatten() with C-order and reshape with F-order, we maintain this
             if phi_data.ndim == 2 and (phi_data.shape[0] == 1 or phi_data.shape[1] == 1) and phi_data.size == 2541:
                  phi_data = phi_data.reshape(121, 21, order='F')
             elif phi_data.ndim == 1 and phi_data.size == 2541:
                  phi_data = phi_data.reshape(121, 21, order='F')

             rows, cols = phi_data.shape
             
             # Physics domain: X from -3m (cathode) to +3m (anode), Y from 0m (surface) to 1m (depth)
             # rows = NX = 121 points in X direction
             # cols = NY = 21 points in Y direction
             x_range = np.linspace(-3.0, 3.0, rows)
             y_range = np.linspace(0.0, 1.0, cols)
                 
             return {
                'phi': phi_data,
                'xpos': x_range,
                'ypos': y_range,
                'currentDensityProfile': compute_current_density(phi_data, x_range, y_range),
                'anodeIndices': np.where(x_range >= 0)[0] + 1,
                'corrosionRate': 0.0 
             }
             
        def compute_current_density(phi, x, y, sigma=4.5):
            # J = - sigma * grad(phi)
            # We are interested in J_y at the surface (assuming surface is at y=0 or y=max)
            # Based on standard BEM, electrolyte is usually y > 0 or y < 0.
            # We'll take gradient along Y dimension.
            
            try:
                # phi shape (row, col) or similar. Check against x, y len.
                if phi.shape == (len(y), len(x)):
                     # y is rows (axis 0), x is cols (axis 1)
                     grad_y, grad_x = np.gradient(phi, y, x)
                     # Take surface current. Assuming surface is Top (index -1) or Bottom (index 0)
                     # In typical galvanic plots data, surface is often index 0 ??
                     # We'll take the max magnitude row to be safe for PoC visualization
                     J_y = -sigma * grad_y
                     
                     # Return the profile along X. 
                     # We return the row with max activity
                     max_idx = np.argmax(np.mean(np.abs(J_y), axis=1))
                     return J_y[max_idx, :]
                elif phi.shape == (len(x), len(y)):
                     # x is rows, y is cols
                     grad_x, grad_y = np.gradient(phi, x, y)
                     J_y = -sigma * grad_y
                     max_idx = np.argmax(np.mean(np.abs(J_y), axis=0))
                     return J_y[:, max_idx]
                else:
                    return np.zeros(len(x))
            except Exception as e:
                print(f"Error computing current density: {e}")
                return np.zeros(len(x))

        # 1. Plot BNN Prediction (Always available)
        if HAS_PLOTTING:
            print("Plotting BNN Prediction...")
            bnn_data = make_bnn_sim_data(mean_field)
            plot_simulation_results(bnn_data, output_dir, run_id="BNN_Prediction")

        # 2. Plot Physics Ground Truth (If available)
        if physics_data is not None:
             if HAS_PLOTTING:
                print("Plotting Physics Ground Truth...")
                plot_simulation_results(physics_data, output_dir, run_id="PHYSICS_GroundTruth")

        print(f"Comparison Plots saved to {output_dir}")
            
    else:
        print("Experiment Failed.")
