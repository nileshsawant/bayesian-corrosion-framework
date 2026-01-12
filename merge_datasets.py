#!/projects/hpcapps/nsawant/corrosion/env/bin/python3
"""
Utility: Merge Multiple Training Datasets
==========================================
Combines multiple .pkl training datasets into one.

Usage:
    python3 merge_datasets.py dataset1.pkl dataset2.pkl merged.pkl
"""

import argparse
import pickle
import numpy as np
import sys


def load_dataset(path):
    """Load a training dataset"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def merge_datasets(datasets):
    """Merge multiple datasets"""
    merged = {
        'inputs': [],
        'outputs': [],
        'metadata': []
    }
    
    total_samples = 0
    
    for i, dataset in enumerate(datasets):
        n_samples = len(dataset['inputs'])
        print(f"  Dataset {i+1}: {n_samples} samples")
        
        merged['inputs'].append(dataset['inputs'])
        merged['outputs'].append(dataset['outputs'])
        merged['metadata'].extend(dataset['metadata'])
        
        total_samples += n_samples
    
    # Concatenate arrays
    merged['inputs'] = np.vstack(merged['inputs'])
    merged['outputs'] = np.vstack(merged['outputs'])
    
    print(f"\nMerged: {total_samples} samples")
    print(f"  Inputs:  {merged['inputs'].shape}")
    print(f"  Outputs: {merged['outputs'].shape}")
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple training datasets',
        epilog="""
Examples:
  # Merge two datasets:
  python3 merge_datasets.py data1.pkl data2.pkl merged.pkl
  
  # Merge many datasets:
  python3 merge_datasets.py batch*.pkl combined.pkl
        """)
    
    parser.add_argument('inputs', nargs='+',
                       help='Input dataset files (.pkl)')
    
    parser.add_argument('output',
                       help='Output merged dataset file')
    
    args = parser.parse_args()
    
    if len(args.inputs) < 2:
        print("ERROR: Need at least 2 input datasets to merge")
        sys.exit(1)
    
    # Separate output from inputs
    input_files = args.inputs[:-1]
    output_file = args.inputs[-1]
    
    # Handle case where output is explicitly last argument
    if len(args.inputs) > 2 and args.output != output_file:
        input_files = args.inputs
        output_file = args.output
    
    print("="*80)
    print("MERGING DATASETS")
    print("="*80)
    print(f"Input files: {len(input_files)}")
    for f in input_files:
        print(f"  - {f}")
    print(f"Output: {output_file}")
    print("="*80)
    print()
    
    # Load datasets
    print("Loading datasets...")
    datasets = []
    for f in input_files:
        try:
            ds = load_dataset(f)
            datasets.append(ds)
        except Exception as e:
            print(f"ERROR loading {f}: {e}")
            sys.exit(1)
    
    # Merge
    print("\nMerging...")
    merged = merge_datasets(datasets)
    
    # Save
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(merged, f)
    
    print("\n" + "="*80)
    print("✓ Merge Complete!")
    print("="*80)
    print(f"Combined dataset: {len(merged['inputs'])} samples")
    print(f"\nNext step: Train on merged dataset")
    print(f"  python3 train_bnn_model.py --data {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
