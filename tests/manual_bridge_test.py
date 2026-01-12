import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from physics_bridge import PhysicsBridge
from plotting_utils import plot_simulation_results

def test_bridge():
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "corrosion-modeling-applications"))
    print(f"Repo path: {repo_path}")
    bridge = PhysicsBridge(repo_path)
    env = [0.01, 20.0, 8.0, 1.0]
    print("Running simulation...")
    res = bridge.run_sim(env)
    print(f"Result: {res['corrosionRate']}")
    
    # Plot results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "test_output")
    plot_simulation_results(res, output_dir, run_id="manual_test")

if __name__ == "__main__":
    test_bridge()
