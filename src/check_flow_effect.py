"""
Check if physics outputs actually vary with flow velocity
"""
import pickle
import numpy as np

# Load training data
with open('../training_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("Checking flow velocity effects in training data...")
print("=" * 60)

# Find pairs where only flow differs
found = 0
for i in range(len(data['inputs'])):
    params_i = data['metadata'][i]['params']
    
    for j in range(i+1, len(data['inputs'])):
        params_j = data['metadata'][j]['params']
        
        # Check if only flow differs
        if (params_i['NaCl'] == params_j['NaCl'] and 
            params_i['Temp'] == params_j['Temp'] and 
            params_i['pH'] == params_j['pH'] and 
            params_i['Flow'] != params_j['Flow']):
            
            # Compare outputs
            output_i = data['outputs'][i]
            output_j = data['outputs'][j]
            
            diff = output_j - output_i
            
            print(f"\nPair {found+1}:")
            print(f"  Sample {i}: Flow={params_i['Flow']:.2f} m/s")
            print(f"  Sample {j}: Flow={params_j['Flow']:.2f} m/s")
            print(f"  Output difference:")
            print(f"    Mean: {np.mean(diff):.6f}")
            print(f"    Std:  {np.std(diff):.6f}")
            print(f"    Max:  {np.max(np.abs(diff)):.6f}")
            print(f"    Min:  {np.min(np.abs(diff)):.6f}")
            
            # Check if outputs are essentially identical
            if np.max(np.abs(diff)) < 1e-6:
                print(f"  ⚠️  OUTPUTS ARE IDENTICAL (diff < 1e-6)")
            else:
                print(f"  ✓ Outputs differ significantly")
            
            found += 1
            if found >= 5:
                break
    
    if found >= 5:
        break

if found == 0:
    print("\n⚠️  No sample pairs found where only flow differs!")
else:
    print(f"\n{'=' * 60}")
    print(f"Analyzed {found} pairs where only flow velocity differs")
