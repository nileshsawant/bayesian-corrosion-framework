# Codebase Cleanup Analysis

## Essential Files (KEEP)

### Core User-Facing Scripts
1. **active_learning.py** ✅ - Main user interface for predictions
2. **generate_training_data.py** ✅ - Generate initial training data
3. **train_bnn_model.py** ✅ - Train/retrain BNN model
4. **merge_datasets.py** ✅ - Utility to combine datasets (useful for users)
5. **generate_flowchart.py** ✅ - Generate documentation diagrams (for pre-proposal)

### Core Library Modules (src/)
1. **src/bnn_model.py** ✅ - BNN implementation (used by active_learning.py)
2. **src/physics_bridge.py** ✅ - Octave interface (used by active_learning.py)
3. **src/train_bnn.py** ✅ - Training logic (used by active_learning.py and train_bnn_model.py)
4. **src/physics_wrapper.m** ✅ - Octave wrapper (called by physics_bridge.py)
5. **src/run_physics.m** ✅ - Physics simulation entry point (called by physics_wrapper.m)

### Essential Data/Models
1. **bnn_model.pt** ✅ - Current trained model
2. **training_data.pkl** ✅ - Current training dataset
3. **active_learning_results/** ✅ - User results (but can clean old ones)

### Essential Documentation
1. **README.md** ✅ - Main documentation
2. **ACTIVE_LEARNING_FLOWCHART.md** ✅ - Workflow documentation (for SERDP)
3. **requirements.txt** ✅ - Dependencies

---

## Files to DELETE - Analysis Scripts (Old Development/Testing)

### Analysis/Debugging Scripts (src/)
- **src/analyze_j_error.py** ❌ - Old error analysis (problem solved)
- **src/demo_j_error.py** ❌ - Old demonstration script (problem solved)
- **src/check_flow_effect.py** ❌ - Old debugging script
- **src/validate_current_density_calc.py** ❌ - Old validation (now working)
- **src/inspect_dataset.py** ❌ - Debug script (users don't need)
- **src/test_current_density.py** ❌ - Old test (problem solved)
- **src/test_simple.py** ❌ - Old test script
- **src/test_flow_direct.py** ❌ - Old test script
- **src/manual_bridge_test.py** ❌ - Manual test (not needed)
- **src/test_training_speed.py** ❌ - Benchmarking script (results known)

### One-Time Migration Scripts (src/)
- **src/regenerate_dataset_with_J.py** ❌ - Migration script (already done)
- **src/retrain_with_J.py** ❌ - Migration script (already done)
- **src/implement_option1.py** ❌ - Implementation experiment (now in active_learning.py)

### Old Architecture Files (src/)
- **src/bnn_model_old.py** ❌ - Old BNN implementation (replaced)
- **src/orchestrator.py** ❌ - Old orchestrator (replaced by active_learning.py)
- **src/generate_dataset.py** ❌ - Old data generation (replaced by generate_training_data.py)
- **src/generate_dataset_test.py** ❌ - Test for old script

### Root-Level Test/Debug Scripts
- **check_torch.py** ❌ - Environment check (not needed)
- **check_oct2py.py** ❌ - Environment check (not needed)
- **compare_env.py** ❌ - Temp script for env comparison
- **debug_bnn.py** ❌ - Debug script
- **run_inference.py** ❌ - Standalone inference (superseded by active_learning.py)

### Test Files
- **tests/fix_octave_classes.py** ❌ - Old fix script
- **tests/test_pipeline.py** ❌ - Old pipeline test
- **tests/test_2d_surrogate.py** ❌ - Old 2D test
- **tests/test_minimal.py** ❌ - Minimal test

---

## Files to DELETE - Documentation

### Redundant/Outdated Documentation
- **WORKFLOW.md** ❌ - Superseded by ACTIVE_LEARNING_FLOWCHART.md
- **CURRENT_DENSITY_ERROR_ANALYSIS.md** ❌ - Historical problem analysis (solved)
- **ENVIRONMENT_STATUS.md** ❌ - Replaced by REQUIREMENTS_COMPARISON.md
- **QUICKSTART.md** ❌ - Redundant with README.md (consolidate if needed)
- **TRAINING_GUIDE.md** ❌ - Redundant with README.md (consolidate if needed)
- **ACTIVE_LEARNING_GUIDE.md** ❌ - Redundant with ACTIVE_LEARNING_FLOWCHART.md

### Keep These Documentation Files
- **README.md** ✅ - Main entry point
- **ACTIVE_LEARNING_FLOWCHART.md** ✅ - Technical workflow (for SERDP)
- **REQUIREMENTS_COMPARISON.md** ✅ - Environment documentation
- **TECHNICAL_NOTES.md** ✅ - Implementation details (useful reference)

---

## Files to DELETE - Old Model Backups

### Old Model Versions (keep only latest backup)
- **bnn_model_old_architecture.pt** ❌ - Old architecture (no longer compatible)
- **bnn_model_phi_only.pt** ❌ - Phi-only model (outdated approach)
- **bnn_model_small.pt** ❌ - Small network (replaced by larger)
- **bnn_model.pt.backup_20260113_105333** ❌ - Old backup (keep only 2-3 most recent)
- (17 other .backup_* files) ❌ - Keep only latest 2-3

### Old Data Backups (keep only latest backup)
- **training_data_phi_only.pkl** ❌ - Phi-only dataset (outdated approach)
- **training_data.pkl.backup_20260113_105333** ❌ - Old backup (keep only 2-3 most recent)
- (17 other .backup_* files) ❌ - Keep only latest 2-3

---

## Directories to CLEAN

### Test Output Directories
- **test_output/** ❌ - Old test outputs
- **test_corrected_axes/** ❌ - Old test outputs
- **test_physics_correct/** ❌ - Old test outputs
- **comparison_plots/** ❌ - Old comparison plots
- **results/** ❌ - Old results (superseded by active_learning_results/)

### Keep These Directories
- **active_learning_results/** ✅ - Current results (clean old entries if needed)
- **src/** ✅ - Core library (after removing obsolete files)
- **data/** ✅ - (check if needed)
- **dependencies/** ✅ - (check if needed)

---

## Log/Temp Files to DELETE
- **check_fix.log** ❌
- **oct_check.log** ❌
- **torch_check.log** ❌
- **test_output.txt** ❌
- **example_active_learning.txt** ❌
- **octave-workspace** ❌ (Octave temp file)
- **src/octave-workspace** ❌ (Octave temp file)
- **src/option1_implementation.log** ❌
- **src/regenerate_log.txt** ❌

---

## Summary

### Files to Keep: ~15 essential files
- 5 user scripts (active_learning, generate_data, train_model, merge_datasets, generate_flowchart)
- 5 core library modules (bnn_model, physics_bridge, train_bnn, 2 .m files)
- 3 documentation files (README, FLOWCHART, REQUIREMENTS_COMPARISON)
- Current model and data files

### Files to Delete: ~50+ obsolete files
- ~15 old test/analysis scripts
- ~5 migration scripts (one-time use)
- ~6 redundant documentation files
- ~20+ old model/data backups (keep only 2-3 recent)
- ~5 test output directories
- ~8 log/temp files

### Recommended Cleanup Commands

```bash
# Remove old test/analysis scripts
rm src/analyze_j_error.py src/demo_j_error.py src/check_flow_effect.py
rm src/validate_current_density_calc.py src/inspect_dataset.py
rm src/test_current_density.py src/test_simple.py src/test_flow_direct.py
rm src/manual_bridge_test.py src/test_training_speed.py

# Remove migration scripts (already executed)
rm src/regenerate_dataset_with_J.py src/retrain_with_J.py src/implement_option1.py

# Remove old architecture files
rm src/bnn_model_old.py src/orchestrator.py src/generate_dataset.py src/generate_dataset_test.py

# Remove root-level test scripts
rm check_torch.py check_oct2py.py compare_env.py debug_bnn.py run_inference.py

# Remove test files
rm tests/fix_octave_classes.py tests/test_pipeline.py tests/test_2d_surrogate.py tests/test_minimal.py

# Remove redundant documentation (REVIEW FIRST to consolidate useful content)
rm WORKFLOW.md CURRENT_DENSITY_ERROR_ANALYSIS.md ENVIRONMENT_STATUS.md
# Consider consolidating QUICKSTART.md, TRAINING_GUIDE.md, ACTIVE_LEARNING_GUIDE.md into README

# Remove old model versions
rm bnn_model_old_architecture.pt bnn_model_phi_only.pt bnn_model_small.pt
rm training_data_phi_only.pkl

# Keep only 3 most recent backups
ls -t bnn_model.pt.backup_* | tail -n +4 | xargs rm
ls -t training_data.pkl.backup_* | tail -n +4 | xargs rm

# Remove old test output directories
rm -rf test_output/ test_corrected_axes/ test_physics_correct/ comparison_plots/ results/

# Remove log/temp files
rm check_fix.log oct_check.log torch_check.log test_output.txt example_active_learning.txt
rm octave-workspace src/octave-workspace src/option1_implementation.log src/regenerate_log.txt
```

---

*Analysis Date: January 14, 2026*
