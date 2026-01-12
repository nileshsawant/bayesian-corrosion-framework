# BNN Training Workflow

## Overview
This framework trains a Bayesian Neural Network (BNN) to predict corrosion potential fields as a surrogate for expensive physics simulations.

## Step-by-Step Workflow

### Step 1: Generate Training Dataset

Run the physics solver over a parameter grid to create training data:

```bash
python bayesian-corrosion-framework/src/generate_dataset.py
```

**Parameter Ranges:**
- **NaCl Concentration**: 0.1 - 1.0 M (4 values)
- **Temperature**: 278 - 313 K (5°C - 40°C) (4 values)
- **pH**: 6.0 - 9.0 (4 values)
- **Flow Velocity**: 0.1 - 3.0 m/s (4 values)

**Total Combinations**: 4×4×4×4 = **256 simulations**

**Output**: `training_data.pkl` (saved in project root)

**Expected Runtime**: ~15-60 minutes per simulation = **64-256 hours total**
(Consider running on HPC with parallel job array)

---

### Step 2: Train the BNN

Train the neural network on the generated dataset:

```bash
python bayesian-corrosion-framework/src/train_bnn.py
```

**What it does:**
- Loads `training_data.pkl`
- Normalizes inputs and outputs
- Trains BNN for 2000 iterations
- Saves trained model to `bnn_model.pt`
- Tests inference on 5 samples

**Expected Runtime**: ~10-30 minutes

---

### Step 3: Use Trained Model for Inference

The trained model can now predict potential fields instantly:

```python
from bnn_model import BNNWrapper
import torch

# Load trained model (automatically loaded)
bnn = BNNWrapper(input_dim=4, output_dim=2541)

# New conditions
params = torch.tensor([[0.6, 298.0, 8.2, 1.0]])  # [NaCl, Temp, pH, Flow]

# Predict with uncertainty
mean, std = bnn.predict(params)

# mean: (1, 2541) - predicted potential field (flattened)
# std: (1, 2541) - epistemic uncertainty per point
```

Reshape to 2D: `phi_2d = mean.reshape(121, 21, order='F')`

---

## Parallel Dataset Generation (HPC)

To speed up dataset generation, use SLURM job arrays:

```bash
# Create job script: generate_data_parallel.sh
#!/bin/bash
#SBATCH --array=0-255
#SBATCH --time=02:00:00
#SBATCH --mem=80G

# Each job runs one simulation
python run_single_sim.py $SLURM_ARRAY_TASK_ID
```

Then combine results:
```bash
python combine_results.py
```

---

## Files

- **`generate_dataset.py`**: Grid sweep over parameters → `training_data.pkl`
- **`train_bnn.py`**: Train BNN on dataset → `bnn_model.pt`
- **`bnn_model.py`**: BNN architecture definition
- **`orchestrator.py`**: (Old) Active learning mode (single-sample)
- **`physics_bridge.py`**: Interface to Octave physics solver
- **`plotting_utils.py`**: Visualization utilities

---

## Current Status

✓ Parameter ranges defined  
✓ Dataset generation script ready  
✓ Training script ready  
⚠ Need to run dataset generation (long runtime)  
⚠ Need to train on generated dataset  

---

## Next Steps

1. **Run dataset generation** (consider HPC parallelization)
2. **Train BNN** on complete dataset
3. **Validate predictions** against held-out test set
4. **Deploy for inference** in your application
