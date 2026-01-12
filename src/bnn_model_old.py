import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO

class PyroLinear(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # He Initialization for Prior Scale: sqrt(2 / in_features)
        # Check for 0 division
        sigma = (2.0 / in_features) ** 0.5
        
        # Define weights and bias as PyroSamples
        # We use a tighter prior to prevent explosion at initialization
        self.weight = PyroSample(dist.Normal(0., sigma).expand([out_features, in_features]).to_event(2))
        self.bias = PyroSample(dist.Normal(0., 0.1).expand([out_features]).to_event(1))

    def forward(self, x):
        w = self.weight
        b = self.bias
        
        # Handle Extra Batch Dims (e.g. from Predictive)
        # If w is (1, Out, In) -> Squeeze it to (Out, In)
        if w.dim() == 3 and w.shape[0] == 1:
            w = w.squeeze(0)
        if b.dim() == 2 and b.shape[0] == 1:
            b = b.squeeze(0)
            
        # Support for true batching if needed later (e.g. num_samples > 1 in parallel)
        # But for now, basic robustness for sequential
        
        return F.linear(x, w, b)

class BayesianCorrosionNet(PyroModule):
    def __init__(self, input_dim=4, output_dim=2541, hidden_dim=64):
        super().__init__()
        
        # Use custom PyroLinear
        self.fc1 = PyroLinear(input_dim, hidden_dim)
        self.fc2 = PyroLinear(hidden_dim, hidden_dim)
        self.fc3 = PyroLinear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.output_dim = output_dim
        
    def forward(self, x, y=None):
        # x shape: (Batch, 4)
        
        # Forward pass through network
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mean = self.fc3(h2) 
        
        # Sample observation noise (global parameter, not per-data)
        # Use tighter prior for stability
        sigma = pyro.sample("sigma", dist.LogNormal(-3.0, 0.5))
        
        # Observation likelihood
        # Use independent for proper broadcasting across output dimensions
        with pyro.plate("data", x.shape[0]):
            # Each output dimension is independent given the network parameters
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
            
        return mean

class BNNWrapper:
    def __init__(self, input_dim=4, output_dim=2541):
        self.model = BayesianCorrosionNet(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize guide properly by creating a prototype trace
        # This ensures AutoDiagonalNormal knows which sites to parameterize
        # Create dummy data for initialization
        dummy_x = torch.zeros(1, input_dim)
        dummy_y = torch.zeros(1, output_dim)
        
        # Create guide - AutoDiagonalNormal will automatically exclude observed sites
        self.guide = AutoDiagonalNormal(self.model)
        
        # Initialize guide by running one training step with dummy data
        # This forces the guide to trace the model properly
        self.optimizer = pyro.optim.Adam({"lr": 0.005})
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())
        
        # Warm up the guide with dummy data
        _ = self.svi.step(dummy_x, dummy_y)
        
    def train_step(self, x_data, y_data):
        # DO NOT clear param store here, it wipes the model!
        # pyro.clear_param_store()
        
        if y_data.dim() == 1:
            y_data = y_data.unsqueeze(0)
            
        # Ensure data shapes are correct
        assert x_data.shape[0] == y_data.shape[0], f"Batch size mismatch: x={x_data.shape[0]}, y={y_data.shape[0]}"
        assert y_data.shape[1] == self.output_dim, f"Output dim mismatch: expected {self.output_dim}, got {y_data.shape[1]}"
            
        # Optimization Loop
        # A single step is not enough to learn a new ground truth
        num_iterations = 500  # Increase iterations for better convergence
        initial_loss = 0
        final_loss = 0
        losses = []
        
        for i in range(num_iterations):
            loss = self.svi.step(x_data, y_data)
            losses.append(loss)
            if i == 0: 
                initial_loss = loss
            final_loss = loss
            
            # Log progress periodically
            if (i + 1) % 100 == 0:
                avg_recent = sum(losses[-100:]) / 100
                print(f"      [Step {i+1}/{num_iterations}] Loss: {avg_recent:.2f}")
            
        print(f"      [BNN Training] Steps: {num_iterations}, Loss: {initial_loss:.2f} -> {final_loss:.2f}")
        return final_loss

    def predict(self, x_input, num_samples=100):
        # Use Predictive with return_sites to get the mean prediction (not obs)
        # We want the network output (mean), not the noisy observations
        predictive = pyro.infer.Predictive(self.model, guide=self.guide, num_samples=num_samples)
        samples = predictive(x_input)
        
        # Get the mean predictions from the network
        # The model returns 'mean' before adding noise
        # We need to get it from the return value or reconstruct it
        
        # Alternative: Sample from guide and run forward passes manually
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample parameters from guide
                guide_trace = pyro.poutine.trace(self.guide).get_trace(x_input, None)
                # Replay those samples in model to get deterministic output
                model_trace = pyro.poutine.trace(
                    pyro.poutine.replay(self.model, trace=guide_trace)
                ).get_trace(x_input, None)
                
                # Get the mean prediction (before adding observation noise)
                # The model returns 'mean' as its output
                mean_pred = model_trace.nodes['_RETURN']['value']
                predictions.append(mean_pred)
        
        predictions = torch.stack(predictions)  # (num_samples, batch, output_dim)
        
        # Compute statistics across samples
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std
