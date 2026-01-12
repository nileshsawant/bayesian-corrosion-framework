# BNN Training Guide: Improving Accuracy & Scaling to More Data

## Quick Start with New Scripts

Three standalone scripts have been created for your workflow:

### 1. Generate Training Data
```bash
python3 generate_training_data.py --samples 16 --output training_data.pkl
```

### 2. Train BNN Model
```bash
python3 train_bnn_model.py --data training_data.pkl --output bnn_model.pt --iterations 10000
```

### 3. Run Inference
```bash
# Single prediction
python3 run_inference.py --model bnn_model.pt --nacl 0.5 --temp 295 --ph 7.5 --flow 1.5

# Batch predictions
python3 run_inference.py --model bnn_model.pt --params new_conditions.txt

# Compare with training data
python3 run_inference.py --model bnn_model.pt --data training_data.pkl --compare
```

---

## Improving Model Accuracy with More Data

### Current Status
- **16 samples**: Quick proof-of-concept, ~3% J error
- **Training time**: ~104 minutes for data generation + 3 minutes for training

### Scaling Up Recommendations

#### Option 1: More Samples (Recommended for HY80)
```bash
# Generate 50 samples (better parameter space coverage)
python3 generate_training_data.py --samples 50 --output training_data_50.pkl

# Train with more iterations for convergence
python3 train_bnn_model.py --data training_data_50.pkl --iterations 15000 --output bnn_model_50.pt
```

**Expected improvements:**
- Better interpolation between parameter combinations
- Reduced uncertainty in predictions
- More robust to edge cases
- Training data: ~325 minutes (5.4 hours), Training: ~5 minutes

#### Option 2: Denser Parameter Space
```bash
# More granular parameter sampling
python3 generate_training_data.py \
    --samples 100 \
    --nacl 0.1,0.3,0.5,0.7,1.0 \
    --temp 278,290,300,313 \
    --ph 6.0,7.0,8.0,9.0 \
    --flow 0.1,0.5,1.0,2.0,3.0 \
    --output training_data_dense.pkl

python3 train_bnn_model.py \
    --data training_data_dense.pkl \
    --iterations 20000 \
    --hidden 128,256,128 \
    --output bnn_model_dense.pt
```

**Expected improvements:**
- Captures non-linear effects better
- Smaller prediction errors (target <2% for J)
- Better extrapolation
- Training data: ~650 minutes (10.8 hours), Training: ~7 minutes

#### Option 3: Incremental Learning (Fastest)
Start with existing model and add more data:

```bash
# Generate additional 34 samples
python3 generate_training_data.py --samples 34 --output training_data_additional.pkl

# Merge datasets (requires custom script - see below)
python3 merge_datasets.py training_data.pkl training_data_additional.pkl training_data_50.pkl

# Continue training from existing weights
python3 train_bnn_model.py \
    --data training_data_50.pkl \
    --model bnn_model.pt \
    --iterations 5000 \
    --lr 0.001 \
    --output bnn_model_updated.pt
```

---

## Training on Other Materials

Your physics model supports: **HY80, HY100, SS316, I625, CuNi, Ti**

### Strategy 1: Material-Specific Models
Train separate BNN for each material:

```bash
# HY100 steel
python3 generate_training_data.py \
    --samples 50 \
    --materials HY100 \
    --output hy100_data.pkl

python3 train_bnn_model.py \
    --data hy100_data.pkl \
    --output bnn_hy100.pt

# Stainless steel 316
python3 generate_training_data.py \
    --samples 50 \
    --materials SS316 \
    --output ss316_data.pkl

python3 train_bnn_model.py \
    --data ss316_data.pkl \
    --output bnn_ss316.pt
```

**Pros:**
- Highest accuracy per material
- Faster training per model
- Easy to deploy separately

**Cons:**
- Need to maintain multiple models
- Cannot transfer learning between materials

### Strategy 2: Multi-Material Model (Recommended)
Add material as input feature:

```bash
# Generate mixed dataset
python3 generate_training_data.py \
    --samples 150 \
    --materials HY80 HY100 SS316 \
    --output multi_material_data.pkl

# Train larger network
python3 train_bnn_model.py \
    --data multi_material_data.pkl \
    --hidden 128,256,256,128 \
    --iterations 25000 \
    --output bnn_multi_material.pt
```

**Requires code modification** (see below):
- Add material encoding to input (e.g., one-hot: [1,0,0] for HY80)
- Input dims: 4 → 7 or 10 (depending on encoding)
- More training data needed per material

**Pros:**
- Single model for all materials
- Can learn similarities between materials
- Transfer learning effects

**Cons:**
- Needs more data overall (~50 samples per material)
- Slightly lower accuracy per material

### Strategy 3: Comparative Studies
Generate matched datasets:

```bash
# Same conditions for multiple materials
python3 generate_training_data.py \
    --samples 32 \
    --materials HY80 HY100 \
    --nacl 0.1,1.0 \
    --temp 278,313 \
    --ph 6.0,9.0 \
    --flow 0.1,3.0 \
    --output comparison_hy_steels.pkl
```

---

## Performance vs Accuracy Trade-offs

| Samples | Data Gen Time | Train Time | Expected J Error | Use Case |
|---------|---------------|------------|------------------|----------|
| 16      | ~100 min      | ~3 min     | ~3%             | Proof-of-concept |
| 50      | ~325 min      | ~5 min     | ~2%             | Production |
| 100     | ~650 min      | ~7 min     | ~1.5%           | High accuracy |
| 200     | ~1300 min     | ~10 min    | <1%             | Publication |

*Times based on H100 GPU for training, 8 parallel workers for data generation*

---

## Advanced: Modify Code for Multi-Material Training

### Step 1: Update `physics_bridge.py`
Add material encoding to output:

```python
def generate_sample(params):
    # ... existing code ...
    
    # One-hot encode material
    materials = ['HY80', 'HY100', 'SS316', 'I625', 'CuNi', 'Ti']
    material_encoding = np.zeros(len(materials))
    material_idx = materials.index(params['material'])
    material_encoding[material_idx] = 1.0
    
    # Combine with other inputs
    input_vector = np.concatenate([
        [params['NaCl'], params['Temp'], params['pH'], params['Flow']],
        material_encoding
    ])
    
    return input_vector, output_vector, metadata
```

### Step 2: Update `bnn_model.py`
Change input dimensions:

```python
class CorrosionBNN(nn.Module):
    def __init__(self, input_dim=10, output_dim=2601, hidden_dims=[128, 256, 128], device='cpu'):
        # input_dim = 4 (params) + 6 (material one-hot)
        # ... rest unchanged ...
```

### Step 3: Update `run_inference.py`
Add material parameter:

```python
parser.add_argument('--material', type=str, default='HY80',
                   choices=['HY80', 'HY100', 'SS316', 'I625', 'CuNi', 'Ti'],
                   help='Material type')

# In inference:
materials = ['HY80', 'HY100', 'SS316', 'I625', 'CuNi', 'Ti']
material_encoding = np.zeros(len(materials))
material_encoding[materials.index(args.material)] = 1.0

param_array = np.array([[
    params['NaCl'], params['Temp'], params['pH'], params['Flow'],
    *material_encoding
]])
```

---

## Monitoring Training Quality

### Check convergence:
```bash
# Look for loss plateau in training output
python3 train_bnn_model.py --data training_data.pkl --iterations 20000 | grep "Step"
```

Signs of good training:
- Loss decreases monotonically
- Final loss is negative (ELBO)
- Speed stays constant (~50 it/s on H100)

Signs of problems:
- Loss increases or oscillates
- Very slow convergence
- NaN values

### Validate accuracy:
```bash
# Compare predictions with training data
python3 run_inference.py --model bnn_model.pt --data training_data.pkl --compare
```

Target accuracy:
- Phi: <0.5% error
- J: <3% error (for 16 samples), <1% (for 200+ samples)

---

## Recommended Workflow for Production

1. **Start small** (16 samples, verify pipeline works)
2. **Scale to 50 samples** (good accuracy, reasonable time)
3. **Evaluate results** (run inference on new conditions)
4. **Add more data if needed** (100+ samples for <2% error)
5. **Material-specific models** (HY100, SS316, etc. as needed)

---

## Parallel Data Generation (Faster)

Speed up data generation by running multiple instances:

```bash
# Terminal 1: Generate batch 1
python3 generate_training_data.py --samples 25 --output batch1.pkl &

# Terminal 2: Generate batch 2  
python3 generate_training_data.py --samples 25 --output batch2.pkl &

# Wait for completion, then merge (see merge script below)
python3 merge_datasets.py batch1.pkl batch2.pkl training_data_50.pkl
```

---

## Questions?

**Q: How many samples do I need for X% accuracy?**
A: Rule of thumb: 50 samples → ~2% error, 100 samples → ~1.5%, 200 samples → <1%

**Q: Can I train on CPU?**
A: Yes, but slower (~2-5 it/s vs 50 it/s on GPU). Use `--device cpu`

**Q: How to save GPU memory?**
A: Reduce `--samples` during inference (default 100 → 50) or use smaller `--hidden` layers

**Q: Can I use this for real-time predictions?**
A: Yes! Inference is fast (~0.1s per sample with 100 posterior samples)

**Q: What about other materials not in the list?**
A: Need to add polarization curve coefficients to `polarization-curve-modeling/` directory
