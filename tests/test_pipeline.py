import sys
import os
import unittest
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator import Orchestrator
from bnn_model import BNNWrapper
from physics_bridge import PhysicsBridge

class TestBayesianFramework(unittest.TestCase):
    
    def setUp(self):
        # Assumes sibling directory structure in workspace
        self.repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "corrosion-modeling-applications"))
        if not os.path.exists(self.repo_path):
            self.skipTest(f"Dependency repo not found at {self.repo_path}")

    def test_bnn_shape(self):
        """Verify BNN accepts inputs and returns mean/std"""
        bnn = BNNWrapper(input_dim=4)
        sample_input = torch.tensor([[0.6, 25.0, 8.0, 2.0]])
        mean, std = bnn.predict(sample_input)
        
        self.assertIsInstance(mean, float)
        self.assertIsInstance(std, float)
        print(f"\nBNN Prediction: {mean} +/- {std}")

    def test_physics_bridge(self):
        """Verify Python can talk to Octave"""
        bridge = PhysicsBridge(self.repo_path)
        # Low severity condition (should be safe/fast)
        env = [0.01, 20.0, 8.0, 1.0] # Cl, T, pH, Flow
        rate = bridge.run_sim(env)
        
        print(f"\nOctave Simulation Result: {rate}")
        self.assertFalse(np.isnan(rate))
        bridge.close()

    def test_orchestrator_integration(self):
        """Verify the full loop"""
        orc = Orchestrator(uncertainty_threshold=0.5, repo_path=self.repo_path)
        
        # Test Input
        env = torch.tensor([[0.6, 25.0, 8.0, 2.0]])
        
        # This should trigger the physics engine (high uncertainty initially)
        result = orc.get_corrosion_rate(env)
        
        print(f"\nOrchestrator Result: {result}")
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
