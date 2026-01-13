# Active Learning Pipeline Guide

## Overview

The active learning pipeline intelligently combines fast BNN predictions with expensive physics simulations based on prediction confidence.

## How It Works

```
┌─────────────────────────────────────────────┐
│  New prediction request                     │
│  (NaCl, Temp, pH, Flow)                    │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Run BNN inference                          │
│  • Get prediction                           │
│  • Calculate uncertainty                    │
└──────────────┬──────────────────────────────┘
               │
         ┌─────┴─────┐
         │           │
    Low  │           │  High
uncertainty         uncertainty
         │           │
         ▼           ▼
┌───────────┐   ┌──────────────┐
│ Use BNN   │   │ Run Physics  │
│ (fast)    │   │ (slow)       │
│ ~0.1s     │   │ ~6.5 min     │
└───────────┘   └──────┬───────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Add to dataset   │
              └────────┬─────────┘
                       │
           ┌───────────▼────────────┐
           │ Accumulated N samples? │
           └───────────┬────────────┘
                       │
                  Yes  │  No
                       ▼
              ┌──────────────────┐
              │ Retrain BNN      │
              │ (fine-tuning)    │
              └──────────────────┘
```

## Quick Start

### Single Prediction

```bash
./active_learning.py \
    --model bnn_model.pt \
    --data training_data.pkl \
    --nacl 0.3 \
    --temp 290 \
    --ph 7.0 \
    --flow 2.5
```

### Batch Predictions

```bash
./active_learning.py \
    --model bnn_model.pt \
    --data training_data.pkl \
    --params example_active_learning.txt \
    --retrain-every 10
```

## Configuration

### Uncertainty Threshold

Controls when to use physics vs BNN:

```bash
# Conservative (more physics simulations, higher accuracy)
./active_learning.py --model bnn_model.pt --uncertainty 0.03  # 3%

# Default (balanced)
./active_learning.py --model bnn_model.pt --uncertainty 0.05  # 5%

# Aggressive (more BNN usage, faster but less accurate)
./active_learning.py --model bnn_model.pt --uncertainty 0.10  # 10%
```

**Recommended values:**
- **0.03 (3%)**: High-stakes applications, need maximum accuracy
- **0.05 (5%)**: Default, good balance
- **0.10 (10%)**: Exploratory analysis, speed prioritized

### Retraining Frequency

How many new samples to accumulate before retraining:

```bash
# Frequent updates (more accurate, more training time)
./active_learning.py --retrain-every 5

# Default
./active_learning.py --retrain-every 10

# Infrequent updates (faster, less overhead)
./active_learning.py --retrain-every 25
```

**Trade-offs:**
- **Lower values (5)**: BNN stays more current, but more training overhead
- **Higher values (25)**: Less training time, but BNN may be outdated longer

## Example Workflow

### Scenario: Exploring New Parameter Space

You want to predict corrosion for 100 new conditions:

```bash
# Create parameter file with 100 conditions
# (some in-distribution, some extrapolation)

./active_learning.py \
    --model bnn_model.pt \
    --data training_data.pkl \
    --params new_exploration.txt \
    --uncertainty 0.05 \
    --retrain-every 10
```

**Expected behavior:**
- ~70 conditions: BNN confident → fast predictions (~0.1s each)
- ~30 conditions: BNN uncertain → physics simulations (~6.5 min each)
- After 10, 20, 30 physics sims: Automatic retraining
- Later predictions: More confident as BNN learns

**Time comparison:**
- All physics: 100 × 6.5 min = **650 minutes**
- Active learning: (70 × 0.002) + (30 × 6.5) = **195 minutes**
- **Speedup: 3.3x** (and BNN is now better trained!)

## Output Statistics

After batch processing, you'll see:

```
==================================================================
ACTIVE LEARNING STATISTICS
==================================================================
Total predictions:     100
BNN used:              70 (70.0%)
Physics used:          30 (30.0%)
Retraining events:     3
Avg BNN uncertainty:   3.24%

Time savings:
  All physics:         650.0 min
  Active learning:     195.1 min
  Speedup:             3.3x
==================================================================
```

## Advanced Usage

### Force Physics Simulation

For testing or validation:

```bash
./active_learning.py \
    --model bnn_model.pt \
    --data training_data.pkl \
    --nacl 0.5 --temp 295 --ph 7.5 --flow 1.5 \
    --force-physics
```

### Custom Dependency Path

If physics code is elsewhere:

```bash
./active_learning.py \
    --model bnn_model.pt \
    --dependency-root /path/to/corrosion-modeling-applications \
    --params batch.txt
```

## Parameter File Format

CSV format: `NaCl(M), Temp(K), pH, Flow(m/s)`

```
# Comments start with #
# In-distribution samples (BNN should be confident)
0.1, 278, 6.0, 0.1
0.5, 295, 7.5, 1.5
1.0, 313, 9.0, 3.0

# Extrapolation samples (BNN should be uncertain)
0.05, 270, 5.0, 0.05
1.5, 320, 10.0, 4.0
```

## When Does Retraining Occur?

Retraining triggers when:
1. Physics simulation completes
2. New sample added to accumulator
3. Accumulator size ≥ `--retrain-every`

**Process:**
1. Backup current dataset → `training_data.pkl.backup_YYYYMMDD_HHMMSS`
2. Backup current model → `bnn_model.pt.backup_YYYYMMDD_HHMMSS`
3. Merge new samples with dataset
4. Fine-tune BNN (5000 iterations, lower learning rate)
5. Save updated dataset and model
6. Reload model
7. Clear accumulator

**Fine-tuning settings:**
- Iterations: 5000 (vs 10000 for initial training)
- Learning rate: 0.001 (vs 0.005 for initial training)
- Time: ~2 minutes on H100 GPU

## Monitoring Progress

During execution, you'll see:

```
======================================================================
Prediction #42
Parameters: NaCl=0.300M, T=290.0K, pH=7.00, Flow=2.500m/s
======================================================================
→ Running BNN inference...
  Relative uncertainty: 2.45%
  Threshold: 5.00%
✓ LOW UNCERTAINTY - Using BNN prediction

----------------------------------------------------------------------
RESULT:
  Source: BNN
  Corrosion Rate: -5.8234e-02 A/m²
  Confidence: 97.6%
----------------------------------------------------------------------
```

Or for uncertain predictions:

```
======================================================================
Prediction #43
Parameters: NaCl=0.050M, T=275.0K, pH=5.50, Flow=0.050m/s
======================================================================
→ Running BNN inference...
  Relative uncertainty: 8.32%
  Threshold: 5.00%
⚠ HIGH UNCERTAINTY - Using physics simulation
  → Running physics simulation...
  ✓ Physics simulation complete
  → New samples collected: 3

----------------------------------------------------------------------
RESULT:
  Source: PHYSICS
  Corrosion Rate: -4.1234e-02 A/m²
----------------------------------------------------------------------
```

## Integration with Existing Scripts

Active learning is **separate from** the standard workflow:

**Standard workflow** (fixed dataset):
```bash
./generate_training_data.py --samples 50
./train_bnn_model.py --data training_data.pkl
./run_inference.py --model bnn_model.pt --params new_conditions.txt
```

**Active learning** (growing dataset):
```bash
# Start with existing model
./active_learning.py \
    --model bnn_model.pt \
    --data training_data.pkl \
    --params new_conditions.txt
    
# Model and dataset automatically updated!
```

## Troubleshooting

**Q: BNN never triggers physics?**
A: Your uncertainty threshold may be too high. Try `--uncertainty 0.03`

**Q: Physics always triggers?**
A: Your threshold may be too low, or you're extrapolating far. Check parameter ranges.

**Q: Retraining takes too long?**
A: Increase `--retrain-every 25` to reduce retraining frequency

**Q: How to disable retraining?**
A: Set `--retrain-every 9999` (effectively disabled) or don't provide `--data`

**Q: Out of memory during retraining?**
A: Dataset is large. Consider splitting or using CPU: edit script to add `device='cpu'` to train_bnn_batch call

## Best Practices

1. **Start with good baseline**: Train initial BNN on 50-100 diverse samples
2. **Tune threshold**: Run small batch first, check BNN vs physics ratio
3. **Monitor statistics**: Review speedup and accuracy trade-offs
4. **Regular backups**: Script auto-backups, but keep your own checkpoints
5. **Validate periodically**: Run known conditions through both BNN and physics to verify
6. **Track uncertainty trends**: If avg uncertainty increases, may need more training data

## Performance Guidelines

| Scenario | Settings | Expected Outcome |
|----------|----------|------------------|
| Interpolation-heavy | `--uncertainty 0.05` | 80-90% BNN usage, 5-10x speedup |
| Mixed | `--uncertainty 0.05` | 60-70% BNN usage, 3-4x speedup |
| Extrapolation-heavy | `--uncertainty 0.03` | 30-50% BNN usage, 1.5-2x speedup |
| High-accuracy needs | `--uncertainty 0.02` | 20-40% BNN usage, but ensures quality |

## Future Enhancements

Potential improvements (not yet implemented):
- Parallel physics simulations (batch processing)
- Adaptive threshold based on recent accuracy
- Active sampling strategies (suggest most informative points)
- Multi-material support
- Integration with optimization loops
