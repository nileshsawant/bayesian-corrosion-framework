import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class CorrosionBNN:
    """
    Simplified Bayesian Neural Network for Corrosion Prediction.
    Uses a cleaner architecture with proper separation of concerns.
    """
    
    def __init__(self, input_dim=4, output_dim=2541, hidden_dims=[64, 128, 64], device='auto'):
        """
        Args:
            input_dim: Number of input features (NaCl, Temp, pH, Flow)
            output_dim: Number of output values (flattened potential field)
            hidden_dims: List of hidden layer sizes
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[BNN] Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"[BNN] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[BNN] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Build layer sizes
        layer_sizes = [input_dim] + hidden_dims + [output_dim]
        self.layer_sizes = layer_sizes
        
        # Input normalization statistics (learned from training data)
        self.input_mean = None
        self.input_std = None
        
        # Output normalization statistics (learned from first training sample)
        self.output_mean = None
        self.output_std = None
        self.is_normalized = False
        
        # Setup optimizer and SVI
        self.optimizer = Adam({"lr": 0.01})
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())
        
        # Clear and initialize Pyro param store for this device
        pyro.clear_param_store()
        pyro.set_rng_seed(42)  # For reproducibility
    
    def normalize_input(self, x):
        """Normalize input to zero mean, unit variance."""
        if self.input_mean is None:
            # First time: compute statistics per feature
            self.input_mean = x.mean(dim=0, keepdim=True).to(self.device)
            self.input_std = (x.std(dim=0, keepdim=True) + 1e-6).to(self.device)
            print(f"      [Input Normalization] Mean: {self.input_mean.squeeze()}, Std: {self.input_std.squeeze()}")
        
        return (x - self.input_mean) / self.input_std
    
    def normalize_output(self, y):
        """Normalize output to zero mean, unit variance."""
        if self.output_mean is None:
            # First time: compute statistics and move to device
            self.output_mean = y.mean().to(self.device)
            self.output_std = (y.std() + 1e-6).to(self.device)  # Add small constant for stability
            self.is_normalized = True
            print(f"      [Output Normalization] Mean: {self.output_mean:.4f}, Std: {self.output_std:.4f}")
        
        return (y - self.output_mean) / self.output_std
    
    def denormalize_output(self, y_norm):
        """Denormalize output back to original scale."""
        if self.output_mean is None:
            return y_norm  # Not yet normalized
        return y_norm * self.output_std + self.output_mean
        
    def model(self, x, y=None):
        """
        Probabilistic model: defines the generative process.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            y: Target tensor (batch_size, output_dim) or None for prediction
        """
        # Ensure tensors are on correct device
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
        
        # Normalize input
        x = self.normalize_input(x)
        
        # Priors for network weights
        # Use He initialization scale for priors
        # Create tensors directly on device for efficiency
        priors = {}
        for i, (in_size, out_size) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            # Weight prior - create directly on device
            weight_prior_std = (2.0 / in_size) ** 0.5
            priors[f'w{i}'] = dist.Normal(
                torch.zeros(out_size, in_size, device=self.device),
                weight_prior_std * torch.ones(out_size, in_size, device=self.device)
            ).to_event(2)
            
            # Bias prior - create directly on device
            priors[f'b{i}'] = dist.Normal(
                torch.zeros(out_size, device=self.device),
                0.1 * torch.ones(out_size, device=self.device)
            ).to_event(1)
        
        # Sample weights from priors
        weights = {}
        for name, prior in priors.items():
            weights[name] = pyro.sample(name, prior)
        
        # Forward pass
        h = x
        for i in range(len(self.hidden_dims)):
            h = torch.mm(h, weights[f'w{i}'].t()) + weights[f'b{i}']
            h = torch.relu(h)
        
        # Output layer (no activation)
        mu = torch.mm(h, weights[f'w{len(self.hidden_dims)}'].t()) + weights[f'b{len(self.hidden_dims)}']
        
        # Separate observation noise for phi and J regions
        # Phi: first 2541 values (121x21 spatial field)
        # J: last 60 values (current density profile)
        phi_len = 121 * 21  # 2541
        
        # LogNormal priors for noise - allow different scales for phi vs J
        sigma_phi = pyro.sample("sigma_phi", dist.LogNormal(0.0, 0.5))
        sigma_j = pyro.sample("sigma_j", dist.LogNormal(0.0, 0.5))
        
        # Split outputs and apply appropriate noise model
        mu_phi = mu[:, :phi_len]
        mu_j = mu[:, phi_len:]
        
        # Likelihood with region-specific noise
        with pyro.plate("data", x.shape[0]):
            if y is not None:
                y_phi = y[:, :phi_len]
                y_j = y[:, phi_len:]
            else:
                y_phi = None
                y_j = None
            
            obs_phi = pyro.sample("obs_phi", dist.Normal(mu_phi, sigma_phi).to_event(1), obs=y_phi)
            obs_j = pyro.sample("obs_j", dist.Normal(mu_j, sigma_j).to_event(1), obs=y_j)
        
        return mu
    
    def guide(self, x, y=None):
        """
        Variational guide: approximates the posterior over weights.
        Uses mean-field approximation (diagonal Gaussian).
        """
        # Normalize input
        x = self.normalize_input(x)
        
        # Create variational parameters for each weight
        # Initialize directly on device for efficiency
        for i, (in_size, out_size) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            # Weight parameters - create directly on device
            w_loc = pyro.param(f'w{i}_loc', 
                             torch.randn(out_size, in_size, device=self.device) * 0.01)
            w_scale = pyro.param(f'w{i}_scale', 
                               0.1 * torch.ones(out_size, in_size, device=self.device),
                               constraint=dist.constraints.positive)
            
            # Sample weight
            pyro.sample(f'w{i}', dist.Normal(w_loc, w_scale).to_event(2))
            
            # Bias parameters - create directly on device
            b_loc = pyro.param(f'b{i}_loc', 
                             torch.randn(out_size, device=self.device) * 0.01)
            b_scale = pyro.param(f'b{i}_scale', 
                               0.1 * torch.ones(out_size, device=self.device),
                               constraint=dist.constraints.positive)
            
            # Sample bias
            pyro.sample(f'b{i}', dist.Normal(b_loc, b_scale).to_event(1))
        
        # Separate sigma parameters for phi and J regions
        # Sigma_phi for potential field (2541 values)
        sigma_phi_loc = pyro.param('sigma_phi_loc', torch.tensor(0.0, device=self.device))
        sigma_phi_scale = pyro.param('sigma_phi_scale', torch.tensor(0.1, device=self.device),
                                      constraint=dist.constraints.positive)
        pyro.sample("sigma_phi", dist.LogNormal(sigma_phi_loc, sigma_phi_scale))
        
        # Sigma_j for current density (60 values)
        sigma_j_loc = pyro.param('sigma_j_loc', torch.tensor(0.0, device=self.device))
        sigma_j_scale = pyro.param('sigma_j_scale', torch.tensor(0.1, device=self.device),
                                    constraint=dist.constraints.positive)
        pyro.sample("sigma_j", dist.LogNormal(sigma_j_loc, sigma_j_scale))
    
    def train_step(self, x, y, num_iterations=1000):
        """
        Train the BNN on a single data point (active learning).
        
        Args:
            x: Input tensor (1, input_dim)
            y: Target tensor (1, output_dim)
            num_iterations: Number of optimization steps
            
        Returns:
            final_loss: Loss after training
        """
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Normalize output for stable training
        y_norm = self.normalize_output(y)
        
        # Use Adam with moderate learning rate for stability
        optimizer = Adam({"lr": 0.005})  # Lower LR for stable convergence
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        losses = []
        
        for i in range(num_iterations):
            loss = svi.step(x, y_norm)
            losses.append(loss)
            
            if (i + 1) % 200 == 0:
                recent_loss = sum(losses[-200:]) / 200
                print(f"      [Step {i+1}/{num_iterations}] Loss: {recent_loss:.2f}")
        
        final_loss = losses[-1]
        print(f"      [Training Complete] Initial Loss: {losses[0]:.2f} -> Final Loss: {final_loss:.2f}")
        
        # Update main SVI optimizer params from scheduled one
        self.svi = svi_scheduled
        
        return final_loss
    
    def predict(self, x, num_samples=100):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            num_samples: Number of posterior samples for prediction
            
        Returns:
            mean: Mean prediction (batch_size, output_dim)
            std: Standard deviation (batch_size, output_dim)
        """
        # Move to device and normalize
        x = x.to(self.device)
        x_norm = self.normalize_input(x)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample from guide
                guide_trace = pyro.poutine.trace(self.guide).get_trace(x_norm, None)
                
                # Get weights from guide samples
                weights = {}
                for name, node in guide_trace.nodes.items():
                    if name.startswith('w') or name.startswith('b'):
                        weights[name] = node['value']
                
                # Forward pass with sampled weights (use normalized input)
                h = x_norm
                for i in range(len(self.hidden_dims)):
                    h = torch.mm(h, weights[f'w{i}'].t()) + weights[f'b{i}']
                    h = torch.relu(h)
                
                # Output (normalized)
                mu_norm = torch.mm(h, weights[f'w{len(self.hidden_dims)}'].t()) + weights[f'b{len(self.hidden_dims)}']
                
                # Denormalize
                mu = self.denormalize_output(mu_norm)
                predictions.append(mu)
        
        # Stack predictions and compute statistics
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch_size, output_dim)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std


class BNNWrapper:
    """
    Wrapper for compatibility with existing orchestrator code.
    """
    
    def __init__(self, input_dim=4, output_dim=2541, hidden_dims=[64, 128, 64], device='auto'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.bnn = CorrosionBNN(input_dim, output_dim, hidden_dims=hidden_dims, device=device)
        
    def train_step(self, x_data, y_data):
        """Train on a single data point."""
        # Ensure correct shapes
        if x_data.dim() == 1:
            x_data = x_data.unsqueeze(0)
        if y_data.dim() == 1:
            y_data = y_data.unsqueeze(0)
            
        return self.bnn.train_step(x_data, y_data, num_iterations=1000)
    
    def predict(self, x_input, num_samples=100):
        """Make predictions with uncertainty."""
        if x_input.dim() == 1:
            x_input = x_input.unsqueeze(0)
            
        return self.bnn.predict(x_input, num_samples=num_samples)
    
    def save(self, path):
        """
        Save model parameters and metadata.
        
        Args:
            path: Path to save model (without extension)
        """
        import pickle
        
        # Save Pyro param store
        pyro.get_param_store().save(path)
        
        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.bnn.hidden_dims,
            'input_mean': self.bnn.input_mean.cpu() if self.bnn.input_mean is not None else None,
            'input_std': self.bnn.input_std.cpu() if self.bnn.input_std is not None else None,
            'output_mean': self.bnn.output_mean.cpu() if self.bnn.output_mean is not None else None,
            'output_std': self.bnn.output_std.cpu() if self.bnn.output_std is not None else None,
            'version': 'v6'
        }
        
        with open(path + '.metadata', 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path, device='auto'):
        """
        Load model from saved parameters.
        
        Args:
            path: Path to saved model (without extension)
            device: Device to load model on
            
        Returns:
            BNNWrapper instance with loaded parameters
        """
        import pickle
        
        # Load metadata
        with open(path + '.metadata', 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        wrapper = cls(
            input_dim=metadata['input_dim'],
            output_dim=metadata['output_dim'],
            device=device
        )
        
        # Restore normalization parameters
        if metadata['input_mean'] is not None:
            wrapper.bnn.input_mean = metadata['input_mean'].to(wrapper.bnn.device)
            wrapper.bnn.input_std = metadata['input_std'].to(wrapper.bnn.device)
        
        if metadata['output_mean'] is not None:
            wrapper.bnn.output_mean = metadata['output_mean'].to(wrapper.bnn.device)
            wrapper.bnn.output_std = metadata['output_std'].to(wrapper.bnn.device)
            wrapper.bnn.is_normalized = True
        
        # Load Pyro param store (handle PyTorch 2.6+ weights_only default)
        import torch
        try:
            pyro.get_param_store().load(path, map_location=wrapper.bnn.device)
        except Exception as e:
            if 'weights_only' in str(e):
                # PyTorch 2.6+ - load manually with weights_only=False
                state_dict = torch.load(path, map_location=wrapper.bnn.device, weights_only=False)
                pyro.get_param_store().set_state(state_dict)
            else:
                raise
        
        return wrapper
