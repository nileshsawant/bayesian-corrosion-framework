"""
Inspect raw dataset to see actual parameter values and outputs
"""
import pickle
import numpy as np

with open('../training_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("Dataset inspection")
print("="*60)
print(f"Total samples: {len(data['inputs'])}")
print()

# Show first 4 samples (should be NaCl=0.1, T=278, pH=6.0 with flow=0.1 and 3.0)
print("First 4 samples:")
for i in range(min(4, len(data['inputs']))):
    inp = data['inputs'][i]
    params = data['metadata'][i]['params']
    phi = data['outputs'][i]
    
    print(f"\nSample {i}:")
    print(f"  Input vector: {inp}")
    print(f"  Params: NaCl={params['NaCl']:.2f}M, T={params['Temp']:.1f}K, pH={params['pH']:.1f}, Flow={params['Flow']:.2f}m/s")
    print(f"  Phi stats: min={phi.min():.6f}, max={phi.max():.6f}, mean={phi.mean():.6f}, std={phi.std():.6f}")
    print(f"  Phi first 5 values: {phi[:5]}")

# Compare samples 0 and 1 in detail
if len(data['outputs']) >= 2:
    print("\n" + "="*60)
    print("Detailed comparison: Sample 0 vs Sample 1")
    print("="*60)
    
    diff = data['outputs'][1] - data['outputs'][0]
    
    print(f"Max absolute difference: {np.max(np.abs(diff)):.15e}")
    print(f"Any non-zero differences? {np.any(diff != 0.0)}")
    
    if np.any(diff != 0.0):
        print(f"Number of non-zero elements: {np.sum(diff != 0.0)} / {len(diff)}")
        print(f"Non-zero differences: {diff[diff != 0.0][:10]}")
    else:
        print("All values are EXACTLY identical (bitwise equal)")
