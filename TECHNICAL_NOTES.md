# Bayesian Corrosion Framework - Technical Documentation

## Model Outputs

The BNN predicts two types of outputs concatenated into a single vector:

### 1. Potential Field (Φ)
- **Dimensions**: 121 × 21 = 2541 values
- **Physical meaning**: Electric potential distribution in the electrolyte
- **Units**: Volts (V)
- **Dependencies**: NaCl concentration, Temperature, pH
- **⚠️ Important**: **Flow velocity does NOT affect the potential field** in this model
  - The potential is determined by Laplace's equation (∇²φ = 0)
  - Boundary conditions depend on electrode potentials (material properties)
  - Flow only affects mass transport, not electrostatic potential

### 2. Current Density Profile
- **Dimensions**: Variable (depends on anode discretization, typically ~61 points)
- **Physical meaning**: Current flux at anode surface
- **Units**: A/m²
- **Calculation**: J = κ × (∇φ) - **Pure post-processing of phi field**
- **⚠️ Important**: **Current density does NOT depend on flow** in this simplified model
  - Calculated via Ohm's law: `J = κ * (phi_surface - phi_above) / dy`
  - Since phi is independent of flow, so is J
  - Conductivity κ and mesh spacing dy are constants

## Model Limitations

This implementation uses a **simplified electrochemical model**:
- **Laplace equation**: ∇²φ = 0 (steady-state, no convection in bulk)
- **Ohm's law**: J = κ × ∇φ (purely resistive)
- **No mass transport**: Concentration fields not computed
- **No boundary layers**: Flow effects not modeled

### What's Missing:
- Convective-diffusion equations (Nernst-Planck)
- Concentration-dependent conductivity
- Butler-Volmer boundary conditions with mass transport
- Boundary layer thickness effects (δ ∝ v^(-1/2))

### Result:
**Flow velocity has ZERO effect on all outputs in this model.**

## Parameter Effects Summary

| Parameter | Affects Φ? | Affects Current Density? | Actual Mechanism in Model |
|-----------|-----------|--------------------------|---------------------------|
| NaCl      | ✓         | ✓                        | Solution conductivity, boundary conditions |
| Temperature | ✓       | ✓                        | Boundary conditions (via Butler-Volmer) |
| pH        | ✓         | ✓                        | Electrode potentials in boundary conditions |
| **Flow velocity** | **✗** | **✗**              | **Not modeled** (would need mass transport equations) |

## Why Keep Flow Velocity as Input?

Even though flow has zero effect in the current simplified model:
1. **Interface consistency** - all environmental parameters present
2. **BNN will learn to ignore it** - weight posterior will show near-zero influence
3. **Future extensibility** - easier to upgrade to full mass-transport model
4. **No harm** - extra input doesn't hurt training (just redundant)

**Expected BNN behavior**: After training, flow parameter weights will be small/insignificant compared to NaCl/T/pH weights.

## Training Data Structure

```python
dataset = {
    'inputs': np.array(shape=(N, 4)),      # [NaCl, Temp, pH, Flow]
    'outputs': np.array(shape=(N, 2541+M)), # [Phi (2541), Current Density (M)]
    'metadata': [
        {
            'params': {'NaCl': float, 'Temp': float, 'pH': float, 'Flow': float},
            'corrosion_rate': float,        # Scalar (A/m²)
            'phi_shape': tuple,             # (121, 21)
            'phi_length': int,              # 2541
            'current_density_length': int,  # M
            'output_length': int            # 2541 + M
        },
        ...
    ]
}
```

## Inference Mode

The trained BNN can predict both outputs simultaneously:

```python
# Input: [NaCl, Temp, pH, Flow]
params = np.array([[0.5, 298.0, 7.5, 1.0]])

# Predict
pred_mean, pred_std = bnn.predict(params)

# Split outputs
phi_length = 2541
phi_pred = pred_mean[:, :phi_length].reshape(121, 21)
current_density_pred = pred_mean[:, phi_length:]

# Uncertainties
phi_uncertainty = pred_std[:, :phi_length].reshape(121, 21)
current_density_uncertainty = pred_std[:, phi_length:]
```

## Physics Model Notes

### Galvanic Corrosion Solver
- **Core**: Laplace equation solver (Jacobi method)
- **Boundary conditions**: Butler-Volmer kinetics at electrodes
- **Assumptions**:
  - Steady-state
  - Dilute solution theory
  - Negligible convection in bulk electrolyte
  - **Flow only affects surface boundary layers**

### Material Properties (CuNi / I625)
- Flow velocity affects:
  - Oxygen reduction reaction (ORR) boundary layer: `delORR = 0.085*(1.0 - v/v0)`
  - Hydrogen evolution reaction (HER) boundary layer: `delHER = 0.15 cm` (constant)
- These boundary layers modify reaction rates → current density
- But do NOT affect bulk potential distribution

## Validation

Expected behavior when changing only flow velocity:
- ✓ Potential field (Φ): **Identical** outputs (model limitation)
- ✓ Current density: **Identical** outputs (post-processed from Φ)
- ✓ Corrosion rate: **Identical** values (averaged current density)
- ✓ BNN training: **Successfully fits data** (MSE → 0 on training set)
- ✓ BNN weights: **Flow parameter has minimal influence** (posterior near prior)

## Version History

- **v1-v4**: Single output (Phi only), parameter order bugs
- **v5**: Correct parameter order [cCl, T, pH, v], added input/output normalization
- **v6**: Dual output (Phi + Current Density), documented flow behavior

## Parameter Order Issues (Resolved)

The codebase has inconsistent parameter order conventions:

1. **PolarizationCurveModel** (polCurveMain.m): Uses `[T, pH, cCl, velocity]`
2. **GalvanicCorrosion** (galvCorrSim.m → butlerVolmer): Expects `[cCl, T, pH, velocity]`

Our wrapper uses **galvanic corrosion order**: `[cCl, T, pH, velocity]` (run_physics.m line 29)

This is **correct** for the BEM solver used in this framework.
