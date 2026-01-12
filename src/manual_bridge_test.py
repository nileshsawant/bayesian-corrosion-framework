import os
import sys

# Ensure current dir is in path (assuming running from bayesian-corrosion-framework root)
sys.path.append(os.getcwd())

try:
    from src.physics_bridge import PhysicsBridge
except ImportError:
    # Try local import if running from src
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from physics_bridge import PhysicsBridge

def main():
    print("Starting manual bridge test (Subprocess Version)...")
    
    # Path to the PATCHED legacy code
    dep_root = '/projects/hpcapps/nsawant/corrosion/corrosion-modeling-applications'
    
    if not os.path.exists(dep_root):
        print(f"Error: Dependency root not found: {dep_root}")
        return

    try:
        bridge = PhysicsBridge(dep_root)
        
        # Test params: [c, T, pH, v]
        # c=0.5, T=25.0, pH=7.0, v=1.0
        params = [0.5, 25.0, 7.0, 1.0]
        
        print(f"Calling run_sim with {params}")
        result = bridge.run_sim(params)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
