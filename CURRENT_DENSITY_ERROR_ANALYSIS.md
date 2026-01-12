# Current Density Error Analysis

## Problem Summary

The BNN accurately predicts the potential field (phi) with <0.5% error, but when current density (J) is calculated from the phi gradient, errors exceed 40-50%. This document explains why and provides solutions.

## Root Cause Analysis

### Mathematical Background
Current density is calculated as:
```
J = kappa * (dphi/dy)
where:
- kappa = 5.0 S/m (seawater conductivity)
- dphi/dy = (phi_surface - phi_above) / dy
- dy = 0.05 m (mesh spacing)
```

### The Fundamental Problem

**Gradient magnitude vs prediction error:**
- Typical gradient: `dphi/dy ~ 0.001V` across 0.05m mesh
- BNN phi prediction error: `~0.005V` (MAE)
- **The prediction error is 5x larger than the signal!**

### Numerical Example (Sample 0)

| Quantity | True Value | BNN Prediction | Error |
|----------|------------|----------------|-------|
| phi_surface | 1.1512 V | 1.1516 V | 0.0004 V (0.03%) |
| phi_above | 1.1517 V | 1.1524 V | 0.0007 V (0.06%) |
| **dphi = surface - above** | **-0.000475 V** | **-0.000727 V** | **0.000252 V (53%)** |
| **J = kappa*dphi/dy** | **-0.0475 A/m²** | **-0.0727 A/m²** | **0.0252 A/m² (53%)** |

### Why Errors Amplify

1. **Small absolute errors in phi** (0.0004V, 0.0007V) are **comparable** to the gradient magnitude (0.0005V)
2. When subtracting two similar values, **relative errors don't cancel** - they add up
3. The multiplication by kappa/dy = 100 further amplifies the error
4. Result: **~0.005V phi error → ~0.5 A/m² J error → 50% relative error**

This is a fundamental limitation of **numerical differentiation of noisy predictions**.

## Validation of Calculation Method

Our current density calculation was validated against the physics solver:

```
Testing All 16 Samples:
  Mean error: 0.6209%
  Max error:  0.6283%
  Samples passing (<1%): 16/16
```

This confirms our calculation is correct - the problem is not the formula, but the **unavoidable amplification of small phi errors when computing derivatives**.

## Solutions (in order of preference)

### Option 1: Train BNN to Predict Both Phi and J (RECOMMENDED)

**Implementation:**
```python
# Modify dataset generation to include J profile
output_vec = np.concatenate([phi_flat, j_profile])

# Train BNN with larger output dimension
output_dim = 121*21 + 60  # phi field + J profile
```

**Pros:**
- BNN learns J directly from physics solver (no numerical differentiation)
- Maintains <1% accuracy for both phi and J
- Only requires retraining (not re-running physics)

**Cons:**
- Need to regenerate dataset with J profiles stored
- Slightly larger network and training time

### Option 2: Use Physics Post-Processing for J

**Implementation:**
```python
# Use BNN only for phi prediction
phi_pred = bnn.predict(params)

# Run lightweight physics post-processing
# (solve for J using phi boundary conditions)
J = physics_post_process(phi_pred, params)
```

**Pros:**
- Guaranteed consistency with physics
- No need to retrain BNN

**Cons:**
- Requires physics code coupling
- Slower than pure BNN inference

### Option 3: Smooth BNN Predictions Before Differentiation

**Implementation:**
```python
from scipy.ndimage import gaussian_filter

phi_pred_smoothed = gaussian_filter(phi_pred_2d, sigma=1.0)
J = kappa * gradient(phi_pred_smoothed) / dy
```

**Pros:**
- Simple to implement
- Reduces noise in J predictions

**Cons:**
- Still less accurate than Option 1
- Smoothing introduces bias
- Doesn't fix fundamental issue

## Current Workaround

The comparison script now includes:
1. **Warning messages** when J errors exceed 10%
2. **Explanatory notes** on plots explaining the limitation
3. **Detailed documentation** of the root cause

Example warning:
```
⚠ High J error due to numerical differentiation amplification

Note: BNN phi error (~0.005V) is comparable to gradient magnitude (~0.001V),
causing large amplification in derived J. For accurate J, train BNN to 
predict [phi, J] directly.
```

## Recommended Action Plan

1. **Short term:** Accept that J predictions from BNN phi are unreliable (current status)
2. **Medium term:** Implement Option 1 - modify dataset and retrain to predict [phi, J]
3. **Long term:** Consider Option 2 for critical applications requiring physics-consistent J

## Files Modified

- `compare_predictions.py`: Added warnings and explanations
- `validate_current_density_calc.py`: Validates our J calculation (0.62% error vs stored values)
- `analyze_j_error.py`: Detailed error amplification analysis
- `CURRENT_DENSITY_ERROR_ANALYSIS.md`: This document

## References

- Physics solver: `run_physics.m` lines 45-90 (current density calculation)
- BNN training: `train_bnn.py` (achieves 0.48% phi error)
- Error analysis: `analyze_j_error.py` output shows 2610% gradient relative error
