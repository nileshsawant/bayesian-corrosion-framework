# BNN Corrosion Modeling - Quick Reference

## Three-Step Workflow

### 1. Generate Data
```bash
./generate_training_data.py --samples 50 --output training_data.pkl
```

### 2. Train Model
```bash
./train_bnn_model.py --data training_data.pkl --output bnn_model.pt --iterations 10000
```

### 3. Run Predictions
```bash
# Single prediction
./run_inference.py --model bnn_model.pt --nacl 0.5 --temp 295 --ph 7.5 --flow 1.5

# Batch from file
./run_inference.py --model bnn_model.pt --params conditions.txt

# Compare with training data
./run_inference.py --model bnn_model.pt --data training_data.pkl --compare
```

---

## Common Tasks

### Improve Model Accuracy
```bash
# Generate more training data (50-100 samples recommended)
./generate_training_data.py --samples 100 --output training_data_large.pkl

# Train with more iterations
./train_bnn_model.py --data training_data_large.pkl --iterations 20000 --output bnn_model_v2.pt
```

### Train on Different Material
```bash
# HY100 steel
./generate_training_data.py --samples 50 --materials HY100 --output hy100_data.pkl
./train_bnn_model.py --data hy100_data.pkl --output bnn_hy100.pt

# Stainless Steel 316
./generate_training_data.py --samples 50 --materials SS316 --output ss316_data.pkl
./train_bnn_model.py --data ss316_data.pkl --output bnn_ss316.pt
```

### Continue Training (Fine-tuning)
```bash
# Add more data
./generate_training_data.py --samples 34 --output additional_data.pkl
./merge_datasets.py training_data.pkl additional_data.pkl training_data_50.pkl

# Continue from existing model
./train_bnn_model.py --data training_data_50.pkl --model bnn_model.pt --iterations 5000 --lr 0.001
```

### Merge Multiple Datasets
```bash
./merge_datasets.py dataset1.pkl dataset2.pkl dataset3.pkl merged.pkl
```

---

## Current Results (HY80, 16 samples)

- **Potential field error**: ~0.27%
- **Current density error**: ~3.17% average (was 50% with old method!)
- **Training time**: 3.2 minutes (10k iterations on H100)
- **Data generation**: 104 minutes (16 samples)

---

## Available Materials

- **HY80** (current)
- **HY100**
- **SS316** (Stainless Steel 316)
- **I625** (Inconel 625)
- **CuNi** (Copper-Nickel alloy)
- **Ti** (Titanium)

---

## Parameter Ranges

| Parameter | Default Range | Units |
|-----------|---------------|-------|
| NaCl      | 0.1 - 1.0     | M (Molarity) |
| Temperature | 278 - 313   | K (278K = 5°C, 313K = 40°C) |
| pH        | 6.0 - 9.0     | - |
| Flow      | 0.1 - 3.0     | m/s |

---

## File Formats

### Parameter File (for batch inference)
```
# NaCl(M), Temp(K), pH, Flow(m/s)
0.1, 278, 6.0, 0.1
0.5, 295, 7.5, 1.5
1.0, 313, 9.0, 3.0
```

### Output Files
- `.pkl` - Training data (pickled Python dict)
- `.pt` - Trained model (PyTorch state dict)
- `.png` - Comparison plots

---

## Help & Documentation

```bash
# Detailed help for each script
./generate_training_data.py --help
./train_bnn_model.py --help
./run_inference.py --help
./merge_datasets.py --help
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed information on:
- Improving accuracy with more data
- Training multi-material models
- Performance tuning
- Advanced workflows

---

## Troubleshooting

**Slow training?**
- Check GPU usage: `nvidia-smi`
- Verify device: Training output should show "cuda" and "H100"

**High errors?**
- Need more training data (try 50-100 samples)
- Increase iterations (--iterations 20000)
- Use larger network (--hidden 128,256,128)

**Out of memory?**
- Reduce posterior samples: `--samples 50` (in inference)
- Use smaller network: `--hidden 32,64,32`

**Long data generation?**
- Expected: ~6.5 min per sample
- 16 samples ≈ 100 min, 50 samples ≈ 325 min
- Run multiple batches in parallel, then merge

---

## Performance Benchmarks

| Dataset Size | Data Gen | Training | Phi Error | J Error |
|--------------|----------|----------|-----------|---------|
| 16 samples   | ~100 min | ~3 min   | 0.27%     | 3.2%    |
| 50 samples   | ~325 min | ~5 min   | 0.2%      | ~2%     |
| 100 samples  | ~650 min | ~7 min   | 0.15%     | ~1.5%   |

*H100 GPU, 8 parallel workers*
