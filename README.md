# Bayesian Corrosion Framework

Bayesian Neural Network (BNN) surrogate model for predicting galvanic corrosion in marine environments. Replaces expensive physics simulations with fast, uncertainty-aware predictions.

## Key Features

- **Direct prediction** of potential fields (phi) and current density (J) profiles
- **3% average error** for corrosion rate predictions (vs 50% with numerical differentiation)
- **Fast inference**: ~0.1s per prediction with uncertainty quantification
- **GPU accelerated**: 50+ iterations/second on H100 GPU
- **Material support**: CuNi, HY80, HY100, SS316, I625, Ti

## Current Performance

- **Potential field**: ~0.27% error
- **Current density**: ~3.17% error (16 training samples)
- **Training time**: ~3 minutes (10k iterations on H100)
- **Data generation**: ~6.5 minutes per sample

## Quick Start

### 1. Generate Training Data
```bash
./generate_training_data.py --samples 50 --output training_data.pkl
```

### 2. Train BNN Model
```bash
./train_bnn_model.py --data training_data.pkl --iterations 10000 --output bnn_model.pt
```

### 3. Run Predictions
```bash
# Single prediction
./run_inference.py --model bnn_model.pt --nacl 0.5 --temp 295 --ph 7.5 --flow 1.5

# Compare with training data
./run_inference.py --model bnn_model.pt --data training_data.pkl --compare
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Command reference and common tasks
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Improving accuracy, multi-material models, scaling strategies
- **[TECHNICAL_NOTES.md](TECHNICAL_NOTES.md)** - Architecture details and implementation notes
- **[CURRENT_DENSITY_ERROR_ANALYSIS.md](CURRENT_DENSITY_ERROR_ANALYSIS.md)** - Problem diagnosis and solution

## Project Structure

```
bayesian-corrosion-framework/
├── generate_training_data.py    # Script 1: Generate physics simulation data
├── train_bnn_model.py            # Script 2: Train BNN on data
├── run_inference.py              # Script 3: Make predictions
├── merge_datasets.py             # Utility: Combine datasets
├── src/
│   ├── bnn_model.py             # Core Bayesian Neural Network
│   ├── train_bnn.py             # Training logic
│   ├── physics_bridge.py        # Interface to physics simulations
│   ├── physics_wrapper.m        # Octave wrapper for legacy code
│   ├── run_physics.m            # Physics simulation entry point
│   └── compare_predictions.py   # Visualization and validation
├── training_data.pkl            # Generated training dataset
├── bnn_model.pt                 # Trained model weights
└── tests/                       # Unit and integration tests
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Pyro (Bayesian framework)
- Octave 8.0+ (for physics simulations)
- CUDA GPU (optional but recommended for training)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

Requires the [Corrosion Modeling Applications](https://github.com/nsawant2024/corrosion-modeling-applications) repository (Octave-compatible fork) for physics simulations.

## Training on Different Materials

Current model trained on: **CuNi (Copper-Nickel)** anode with I625 cathode

To train on other materials:
```bash
# HY80 steel
./generate_training_data.py --samples 50 --materials HY80 --output hy80_data.pkl
./train_bnn_model.py --data hy80_data.pkl --output bnn_hy80.pt

# Stainless Steel 316
./generate_training_data.py --samples 50 --materials SS316 --output ss316_data.pkl
./train_bnn_model.py --data ss316_data.pkl --output bnn_ss316.pt
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for multi-material models.

## Parameter Ranges

| Parameter | Range | Unit |
|-----------|-------|------|
| NaCl      | 0.1 - 1.0 | M (Molarity) |
| Temperature | 278 - 313 | K (5-40°C) |
| pH        | 6.0 - 9.0 | - |
| Flow velocity | 0.1 - 3.0 | m/s |

## Key Innovation: Direct J Prediction

Traditional approach: Train BNN on phi → Calculate J = -κ∇φ
- **Problem**: Numerical differentiation amplifies small errors
- **Result**: 50% error in J despite <1% error in phi

**Our solution**: Train BNN to predict [phi, J] concatenated
- **Result**: 3% error in J, matching phi accuracy
- **Method**: Physics simulations provide both phi field and J profile

## Performance Scaling

| Dataset Size | Data Gen Time | Training Time | Expected J Error |
|--------------|---------------|---------------|------------------|
| 16 samples   | ~100 min      | ~3 min        | ~3%             |
| 50 samples   | ~325 min      | ~5 min        | ~2%             |
| 100 samples  | ~650 min      | ~7 min        | ~1.5%           |
| 200 samples  | ~1300 min     | ~10 min       | <1%             |

*Based on H100 GPU for training, 8 parallel workers for data generation*

## Citation

If you use this code, please cite:
```
[Citation to be added]
```

## License

[License to be determined]

## Acknowledgments

- Based on legacy MATLAB corrosion modeling code (converted to Octave)
- Bayesian framework built with Pyro
- GPU optimization on H100 GPU at [HPC cluster]
