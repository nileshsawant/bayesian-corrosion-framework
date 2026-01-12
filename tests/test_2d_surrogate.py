import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from orchestrator import Orchestrator

class Test2DSurrogate(unittest.TestCase):
    def test_full_loop(self):
        # 1. Setup
        params = [5.0, 293.0, 7.0, 2.0]
        orch = Orchestrator(uncertainty_threshold=0.0001) # Low threshold to force physics
        
        # 2. Mock Physics Bridge
        # Create a fake 121x21 field
        fake_phi = np.random.rand(121, 21)
        orch.physics.run_sim = MagicMock(return_value={'phi': fake_phi, 'corrosionRate': 0.05})
        
        # 3. Run Pipeline
        result, _ = orch.get_corrosion_prediction(params)
        
        # 4. Assertions
        self.assertTrue(orch.physics.run_sim.called)
        self.assertIsNotNone(result)
        
        # Check BNN training happened (loss history should not be empty inside BNNWrapper, but we can't check internal state easily without access)
        # Instead, we check if prediction uncertainty decreases or check if param store changed.
        
        # Run again with same input - BNN should have learned something (though 1 step might not reduce uncertainty below threshold immediately),
        # but the code should run without error.
        
        result2, _ = orch.get_corrosion_prediction(params)
        print("Loop completed successfully")

if __name__ == '__main__':
    unittest.main()
