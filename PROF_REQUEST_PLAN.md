# Professor Hait's Request Implementation Plan

## Request Analysis

**Key Insight**: "τ" (tau/saturation threshold) = Best value 0.9 for GP fusion operators
- Table 17 already computes extended metrics (AUC, MCC, ROC) with τ ∈ {0.6, 0.7, 0.8, 0.9}
- Need to fix τ=0.9 and vary other parameters to plot their effect

## Requirements

### 1. Extended Metrics (Already Implemented in Table 17)
- ✓ Accuracy, Sensitivity, Specificity, AUC, F1, MCC
- ✓ ROC curves
- ✓ Confusion matrices

### 2. Parameter Variation Plots (τ=0.9 fixed)

**A. Feature Number vs. Accuracy**
- Generate different numbers of features using Adaptive GA with different target sizes
- Evaluate with DenseNet121+SE, apply GP fusion with τ=0.9
- Plot: x-axis = #selected features, y-axis = accuracy (with all extended metrics)

**B. Generation Number vs. Accuracy**
- Run Adaptive GA with n_gen ∈ {10, 20, 30, 50, 80, 100}
- Same setup: DenseNet121+SE → Adaptive GA → classifiers → GP fusion τ=0.9
- Plot: x-axis = n_generations, y-axis = accuracy

**C. Mutation Rate vs. Accuracy**
- Run Adaptive GA with fixed mut_prob ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.3}
- Same setup, τ=0.9
- Plot: x-axis = mutation probability, y-axis = accuracy

**D. Crossover Rate vs. Accuracy**
- Run Adaptive GA with fixed cx_prob ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
- Same setup, τ=0.9
- Plot: x-axis = crossover probability, y-axis = accuracy

### 3. Attention Module Comparison
- DenseNet121 + each attention mechanism (SE, ECA, CBAM, Split, Dual, ViT, Swin)
- Run Adaptive GA + classifiers + GP fusion τ=0.9
- Extended metrics for each
- Identify best performer

## Implementation

**New Experiment Script**: `experiments/table18_parameter_ablation.py`
- Runs all 4 parameter variation experiments with τ=0.9
- Generates plots and CSV files
- Includes attention comparison with extended metrics

**Output Files**:
- `results/table18_feature_number_vs_accuracy.csv`
- `results/table18_generation_vs_accuracy.csv`
- `results/table18_mutation_vs_accuracy.csv`
- `results/table18_crossover_vs_accuracy.csv`
- `results/table18_attention_extended.csv`
- `figures/` - 4 plots + ROC curves per experiment

## Status
- [ ] Create table18_parameter_ablation.py
- [ ] Create table19_attention_extended.py (optional - or combine with 18)
- [ ] Test run locally on small subset
- [ ] Run on cloud
- [ ] Generate figures directory
- [ ] Create combined LaTeX

## Note on Existing Tables
- Table 1: τ variation already shows τ=0.9 is best
- Table 4: Cross-validation (10-fold) already done
- Table 5: Attention comparison already done (but without τ=0.9 GP fusion)
- Table 11-14: Ablation studies already done
- Table 17: Saturation threshold with extended metrics already done

The key gap is combining the ablation parameters WITH τ=0.9 and generating PLOTS.
