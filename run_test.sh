#!/bin/bash
#SBATCH --job-name=bnn_test
#SBATCH --account=hdcomb
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Quick test: 2x2x2x2 = 16 samples
# Should complete in ~30-60 minutes

echo "========================================="
echo "BNN Pipeline Test"
echo "========================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo "========================================="

# Activate environment
source env/bin/activate

# Step 1: Generate small test dataset (2 values per parameter = 16 samples)
echo ""
echo "Step 1: Generating test dataset (16 samples)..."
python bayesian-corrosion-framework/src/generate_dataset_test.py

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset generation failed!"
    exit 1
fi

# Step 2: Train BNN on test dataset
echo ""
echo "Step 2: Training BNN on test dataset..."
python bayesian-corrosion-framework/src/train_bnn.py

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "Test Complete! Check results/"
echo "========================================="
