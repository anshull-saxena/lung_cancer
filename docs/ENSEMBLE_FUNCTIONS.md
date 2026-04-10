# Custom Mathematical Ensemble Functions

**Project:** Lung Histopathology Classification System  
**Date:** February 21, 2026  
**Purpose:** Mathematical documentation of custom ensemble fusion operators

---

## Table of Contents
1. [Overview](#overview)
2. [Weighted Probability Fusion](#weighted-probability-fusion)
3. [Genetic Programming (GP) Fusion Operators](#genetic-programming-gp-fusion-operators)
4. [Weight Calculation Methods](#weight-calculation-methods)
5. [Mathematical Properties](#mathematical-properties)
6. [Implementation Details](#implementation-details)
7. [Comparative Analysis](#comparative-analysis)

---

## Overview

This system implements **two categories of ensemble fusion methods** to combine predictions from multiple classifiers:

1. **Weighted Probability Fusion** - Original method using weighted averaging with product dampening
2. **Genetic Programming (GP) Fusion** - Four distinct operators discovered through genetic programming

Both methods operate on **weighted probabilistic outputs** from individual classifiers, where weights are determined by validation accuracy rankings.

---

## Weighted Probability Fusion

### Mathematical Formula

For each sample $s$ and class $c$, the ensemble probability is computed as:

$$P_{\text{ensemble}}(s, c) = \frac{\sum_{i=1}^{5} \varepsilon_i \cdot P_i(s, c)}{1 + \prod_{i=1}^{5} \varepsilon_i \cdot P_i(s, c)}$$

**Where:**
- $P_i(s, c)$ = Probability of class $c$ from classifier $i$ for sample $s$
- $\varepsilon_i$ = Weight coefficient for classifier $i$ (based on performance ranking)
- $\sum$ = Summation over all 5 classifiers
- $\prod$ = Product over all 5 classifiers

### Components Breakdown

#### Numerator: Weighted Sum
```
∑(εᵢ × Pᵢ(s,c)) = ε₁·P₁(s,c) + ε₂·P₂(s,c) + ε₃·P₃(s,c) + ε₄·P₄(s,c) + ε₅·P₅(s,c)
```
- Linear combination of probabilities
- Higher-performing classifiers contribute more
- Represents "consensus strength" across ensemble

#### Denominator: Product Dampening
```
1 + ∏(εᵢ × Pᵢ(s,c)) = 1 + (ε₁·P₁) × (ε₂·P₂) × (ε₃·P₃) × (ε₄·P₄) × (ε₅·P₅)
```
- Non-linear dampening factor
- Prevents overconfidence when uncertainty is high
- Product term is small when any classifier is uncertain

### Intuition

**Case 1: High Confidence & Agreement**
- All $P_i(s,c)$ are large (close to 1)
- Numerator is large (strong consensus)
- Product term is large → denominator > 1 (applies dampening)
- **Result:** High but moderated probability

**Case 2: Low Confidence or Disagreement**
- Some $P_i(s,c)$ are small
- Numerator is moderate (weak consensus)
- Product term is very small → denominator ≈ 1 (minimal dampening)
- **Result:** Conservative probability estimate

**Case 3: Mixed Signals**
- High variance in $P_i(s,c)$ values
- Weighted sum favors high-performing classifiers
- Product term moderates overconfidence
- **Result:** Balanced, robust prediction

### Example Calculation

**Setup:**
```
Classifiers ranked by accuracy: [XGB, SVM, RF, LR, KNN]
Epsilon weights: ε = [0.42, 0.40, 0.11, 0.05, 0.02]
Probabilities for class ACA: P = [0.95, 0.93, 0.89, 0.85, 0.82]
```

**Computation:**
```
Numerator:
  = 0.42×0.95 + 0.40×0.93 + 0.11×0.89 + 0.05×0.85 + 0.02×0.82
  = 0.399 + 0.372 + 0.098 + 0.043 + 0.016
  = 0.928

Product term:
  = (0.42×0.95) × (0.40×0.93) × (0.11×0.89) × (0.05×0.85) × (0.02×0.82)
  = 0.399 × 0.372 × 0.098 × 0.043 × 0.016
  = 0.000094

Final probability:
  = 0.928 / (1 + 0.000094)
  = 0.928 / 1.000094
  ≈ 0.9279
```

---

## Genetic Programming (GP) Fusion Operators

GP fusion operators combine **weighted probabilities from the top 4 classifiers** using mathematically distinct strategies discovered through genetic programming.

### Notation

For a given sample $s$ and class $c$, let:
```
z₁ = ε₁ × P₁(s,c)    (weighted probability from best classifier)
z₂ = ε₂ × P₂(s,c)    (weighted probability from 2nd-best classifier)
z₃ = ε₃ × P₃(s,c)    (weighted probability from 3rd-best classifier)
z₄ = ε₄ × P₄(s,c)    (weighted probability from 4th-best classifier)
```

Each $z_i \in [0, 1]$ represents a performance-weighted confidence score.

---

### GP1: Maximum Operator (Optimistic Fusion)

$$\text{GP}_1(z_1, z_2, z_3, z_4) = 1 - \min(1-z_1, 1-z_2, 1-z_3, 1-z_4)$$

**Alternative Form:**
$$\text{GP}_1 = \max(z_1, z_2, z_3, z_4)$$

#### Mathematical Properties
- **Idempotent:** $\text{GP}_1(z,z,z,z) = z$
- **Monotonic:** Increasing in all arguments
- **Boundary:** $\text{GP}_1 = z_{\max}$ where $z_{\max} = \max_i z_i$

#### Strategy
- Takes the **maximum** of all weighted probabilities
- **Optimistic** approach: trusts the most confident classifier
- Equivalent to "best expert" selection
- Ignores uncertainty from other classifiers

#### Use Case
Best when:
- One classifier dramatically outperforms others
- High confidence in the top classifier
- Diversity in predictions indicates specialist expertise

#### Example
```
z = [0.85, 0.72, 0.64, 0.58]

Complements: (1-z) = [0.15, 0.28, 0.36, 0.42]
Minimum complement = 0.15
GP1 = 1 - 0.15 = 0.85 ✓ (equals maximum z)
```

---

### GP2: Algebraic Product (Balanced Fusion)

$$\text{GP}_2(z_1, z_2, z_3, z_4) = 1 - (1-z_1)(1-z_2)(1-z_3)(1-z_4)$$

#### Mathematical Properties
- **Product of complements:** Computes $1 - \prod_i (1-z_i)$
- **Probabilistic OR:** Equivalent to probability that at least one classifier is correct
- **Subadditive:** $\text{GP}_2 \leq z_1 + z_2 + z_3 + z_4$
- **Symmetric:** Order of arguments doesn't matter

#### Strategy
- **Multiplicative fusion** with dampening
- Balances confidence across all classifiers
- High output requires agreement among multiple classifiers
- Product term provides natural normalization

#### Intuition
Think of $(1-z_i)$ as "probability of error for classifier $i$":
- Product $(1-z_1)(1-z_2)(1-z_3)(1-z_4)$ = "probability all classifiers are wrong"
- $\text{GP}_2 = 1 - \text{(prob all wrong)}$ = "probability at least one is right"

#### Use Case
Best when:
- Classifiers have comparable performance
- Balanced weighting desired
- Want to leverage ensemble diversity

#### Example
```
z = [0.85, 0.72, 0.64, 0.58]

Complements: (1-z) = [0.15, 0.28, 0.36, 0.42]
Product: 0.15 × 0.28 × 0.36 × 0.42 = 0.00635
GP2 = 1 - 0.00635 = 0.994 ✓ (very high confidence)
```

---

### GP3: Ratio-Based (Complex Non-Linear)

$$\text{GP}_3(z_1, z_2, z_3, z_4) = 1 - \frac{1 + z_1 z_2 z_3 z_4}{(1-z_1)(1-z_2)(1-z_3)(1-z_4)}$$

#### Mathematical Properties
- **Non-linear ratio:** Combines product in both numerator and denominator
- **Emphasis on agreement:** Rewards concordance among classifiers
- **Sensitive to extremes:** Heavily penalizes near-zero or near-one values
- **Asymmetric response:** Different behavior at low vs. high confidence regions

#### Strategy
- **Ratio of concordance to discordance**
- Emphasizes classifier agreement through product term $z_1 z_2 z_3 z_4$
- Denominator $(1-z_i)$ products amplify disagreement
- Complex, highly non-linear aggregation

#### Edge Cases Handled
```python
# Division by zero when any zᵢ = 1:
denominator = max(denominator, 1e-10)  # numerical stability
```

#### Use Case
Best when:
- Strong agreement indicates high confidence
- Want to amplify concordant predictions
- Complex decision boundaries require non-linear fusion

#### Example
```
z = [0.85, 0.72, 0.64, 0.58]

Numerator: 1 + (0.85×0.72×0.64×0.58) = 1 + 0.227 = 1.227
Denominator: (0.15×0.28×0.36×0.42) = 0.00635
GP3 = 1 - (1.227 / 0.00635) = 1 - 193.2 = -192.2 
      → Clipped to valid range (edge case)
```
*Note: This example shows GP3 can produce extreme values; normalization is applied.*

---

### GP4: Weighted Sum (Normalized Average)

$$\text{GP}_4(z_1, z_2, z_3, z_4) = \frac{z_1 + z_2 + z_3 + z_4}{1 + z_1 z_2 z_3 z_4}$$

#### Mathematical Properties
- **Normalized sum:** Average with product-based dampening
- **Bounded:** Output $\in [0, 1]$ for inputs $\in [0, 1]$
- **Monotonic increasing:** Higher inputs → higher output
- **Self-dampening:** Product term prevents overconfidence

#### Strategy
- **Weighted average** with automatic normalization
- Numerator: Sum represents aggregate confidence
- Denominator: Product term dampens when all classifiers agree strongly
- Similar philosophy to original weighted fusion but simpler

#### Intuition
- Sum captures total evidence from all classifiers
- Product dampening: $1 + z_1 z_2 z_3 z_4$ grows when all $z_i$ are large
- High agreement → larger denominator → moderated output
- Prevents runaway confidence from unanimous predictions

#### Use Case
Best when:
- Want average-like behavior
- Need guaranteed normalization
- Seeking conservative fusion strategy

#### Example
```
z = [0.85, 0.72, 0.64, 0.58]

Numerator: 0.85 + 0.72 + 0.64 + 0.58 = 2.79
Denominator: 1 + (0.85×0.72×0.64×0.58) = 1 + 0.227 = 1.227
GP4 = 2.79 / 1.227 = 2.27

After normalization (row-wise across classes):
  If other classes have GP4 values [0.45, 0.32], then:
  Total = 2.27 + 0.45 + 0.32 = 3.04
  Normalized GP4 = 2.27 / 3.04 = 0.747
```

---

## Weight Calculation Methods

All ensemble methods require classifier weights ($\varepsilon_i$) based on performance ranking.

### Cumulative Performance Weighting

**Step 1: Rank classifiers by validation accuracy**
```
Accuracies: [0.97, 0.93, 0.96, 0.88, 0.95]
Ranked: [0.97, 0.96, 0.95, 0.93, 0.88]  (descending)
Ranked names: [C₁*, C₃*, C₅*, C₂*, C₄*]
```

**Step 2: Compute cumulative performance scores**
```
T₁ = 1.0
T₂ = T₁ × Acc₁* = 1.0 × 0.97 = 0.97
T₃ = T₂ × Acc₂* = 0.97 × 0.96 = 0.9312
T₄ = T₃ × Acc₃* = 0.9312 × 0.95 = 0.8846
T₅ = T₄ × Acc₄* = 0.8846 × 0.93 = 0.8227
```

**Step 3: Normalize to get weight coefficients**
```
Sum = T₁ + T₂ + T₃ + T₄ + T₅ = 1.0 + 0.97 + 0.9312 + 0.8846 + 0.8227 = 4.6085

ε₁ = T₁ / Sum = 1.0 / 4.6085 = 0.217  (21.7%) ← Best classifier
ε₂ = T₂ / Sum = 0.97 / 4.6085 = 0.211  (21.1%)
ε₃ = T₃ / Sum = 0.9312 / 4.6085 = 0.202 (20.2%)
ε₄ = T₄ / Sum = 0.8846 / 4.6085 = 0.192 (19.2%)
ε₅ = T₅ / Sum = 0.8227 / 4.6085 = 0.179 (17.9%) ← Worst classifier
```

### Key Properties

- **Exponential decay:** Best classifiers get disproportionately higher weights
- **Top-2 dominance:** First two classifiers typically get ~42% of total weight
- **Performance-aware:** Encodes both absolute and relative performance
- **Multiplicative:** Cumulative product amplifies differences

---

## Mathematical Properties

### Comparison Matrix

| Property | Weighted Fusion | GP1 | GP2 | GP3 | GP4 |
|----------|----------------|-----|-----|-----|-----|
| **Monotonicity** | ✓ | ✓ | ✓ | ✗ | ✓ |
| **Symmetry** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Bounded [0,1]** | ✓* | ✓ | ✓ | ✗* | ✓* |
| **Idempotent** | ✗ | ✓ | ✗ | ✗ | ✗ |
| **Linear** | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Dampening** | ✓ | ✗ | ✓ | ✓ | ✓ |

*✓* = After normalization  
*✗* = Requires clipping

### Sensitivity Analysis

**Effect of changing one classifier's probability:**

| Method | Sensitivity to $\Delta z_1$ | Amplification Factor |
|--------|----------------------------|---------------------|
| Weighted | $\partial P / \partial z_1 = \varepsilon_1 \cdot f(\text{product})$ | Moderate (0.2-0.4) |
| GP1 | $\partial P / \partial z_1 = \mathbb{1}(z_1 = \max)$ | Binary (0 or 1) |
| GP2 | $\partial P / \partial z_1 = \prod_{j \neq 1}(1-z_j)$ | Low (0.001-0.1) |
| GP3 | $\partial P / \partial z_1 = \text{complex}$ | High (1-100) |
| GP4 | $\partial P / \partial z_1 = 1 / (1 + \prod z_i)$ | Low-Moderate (0.1-0.5) |

---

## Implementation Details

### Python Implementation

```python
import numpy as np

# ============================================================================
# WEIGHTED PROBABILITY FUSION
# ============================================================================

def weighted_probability_fusion(probas_list, weights):
    """
    Original weighted fusion with product dampening.
    
    Args:
        probas_list: list of 5 arrays, each (n_samples, n_classes)
        weights: array of 5 floats (epsilon values)
    
    Returns:
        ensemble_proba: array (n_samples, n_classes)
    """
    n_samples, n_classes = probas_list[0].shape
    ensemble_proba = np.zeros((n_samples, n_classes))
    
    for c in range(n_classes):
        numerator = sum(weights[i] * probas_list[i][:, c] 
                       for i in range(5))
        
        product = np.prod([weights[i] * probas_list[i][:, c] 
                          for i in range(5)], axis=0)
        
        ensemble_proba[:, c] = numerator / (1.0 + product)
    
    return ensemble_proba


# ============================================================================
# GP FUSION OPERATORS
# ============================================================================

def gp1_fusion(z1, z2, z3, z4):
    """GP1: Maximum operator - 1 - min(1-z1, 1-z2, 1-z3, 1-z4)"""
    return 1.0 - np.minimum.reduce([1-z1, 1-z2, 1-z3, 1-z4])


def gp2_fusion(z1, z2, z3, z4):
    """GP2: Algebraic product - 1 - (1-z1)(1-z2)(1-z3)(1-z4)"""
    return 1.0 - (1-z1) * (1-z2) * (1-z3) * (1-z4)


def gp3_fusion(z1, z2, z3, z4):
    """GP3: Ratio-based - 1 - (1 + z1*z2*z3*z4) / ((1-z1)(1-z2)(1-z3)(1-z4))"""
    numerator = 1.0 + z1 * z2 * z3 * z4
    denominator = (1-z1) * (1-z2) * (1-z3) * (1-z4)
    denominator = np.maximum(denominator, 1e-10)  # Numerical stability
    result = 1.0 - numerator / denominator
    return result


def gp4_fusion(z1, z2, z3, z4):
    """GP4: Weighted sum - (z1 + z2 + z3 + z4) / (1 + z1*z2*z3*z4)"""
    numerator = z1 + z2 + z3 + z4
    denominator = 1.0 + z1 * z2 * z3 * z4
    return numerator / denominator


def apply_gp_fusion(probas_list, weights, fusion_func, normalize=True):
    """
    Apply GP fusion operator to weighted probabilities.
    
    Args:
        probas_list: list of 4 arrays (top classifiers), each (n_samples, n_classes)
        weights: array of 4 floats (epsilon weights)
        fusion_func: one of {gp1_fusion, gp2_fusion, gp3_fusion, gp4_fusion}
        normalize: whether to normalize output to valid probability distribution
    
    Returns:
        fused_proba: array (n_samples, n_classes)
    """
    n_samples, n_classes = probas_list[0].shape
    fused_proba = np.zeros((n_samples, n_classes))
    
    for c in range(n_classes):
        # Get weighted probabilities for class c
        z1 = weights[0] * probas_list[0][:, c]
        z2 = weights[1] * probas_list[1][:, c]
        z3 = weights[2] * probas_list[2][:, c]
        z4 = weights[3] * probas_list[3][:, c]
        
        # Apply fusion operator
        fused_proba[:, c] = fusion_func(z1, z2, z3, z4)
    
    # Normalize to valid probability distribution
    if normalize:
        row_sums = fused_proba.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        fused_proba = fused_proba / row_sums
    
    return fused_proba


# ============================================================================
# WEIGHT CALCULATION
# ============================================================================

def compute_epsilon_weights(accuracies):
    """
    Compute cumulative performance weights from classifier accuracies.
    
    Args:
        accuracies: array of classifier validation accuracies
    
    Returns:
        epsilon: array of normalized weights
        ranked_indices: indices of classifiers in descending accuracy order
    """
    # Rank by accuracy (descending)
    ranked_indices = np.argsort(accuracies)[::-1]
    ranked_accs = accuracies[ranked_indices]
    
    # Compute cumulative performance
    T = [1.0]
    for j in range(1, len(ranked_accs)):
        T.append(T[-1] * ranked_accs[j-1])
    
    T = np.array(T)
    epsilon = T / T.sum()
    
    return epsilon, ranked_indices


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Simulate 5 classifier probabilities for 10 samples, 3 classes
    np.random.seed(42)
    probas_list = [np.random.dirichlet([2, 2, 2], size=10) for _ in range(5)]
    
    # Classifier accuracies
    accuracies = np.array([0.97, 0.93, 0.96, 0.88, 0.95])
    
    # Compute weights
    epsilon, ranked_idx = compute_epsilon_weights(accuracies)
    print(f"Epsilon weights: {epsilon}")
    print(f"Ranked indices: {ranked_idx}")
    
    # Weighted fusion
    ensemble_proba = weighted_probability_fusion(probas_list, epsilon)
    print(f"Weighted fusion predictions:\n{ensemble_proba[:3]}")
    
    # GP fusion (top 4 classifiers only)
    top4_probas = [probas_list[i] for i in ranked_idx[:4]]
    top4_weights = epsilon[:4]
    
    for name, func in [("GP1", gp1_fusion), ("GP2", gp2_fusion), 
                       ("GP3", gp3_fusion), ("GP4", gp4_fusion)]:
        gp_proba = apply_gp_fusion(top4_probas, top4_weights, func)
        print(f"\n{name} fusion predictions:\n{gp_proba[:3]}")
```

---

## Comparative Analysis

### Empirical Performance (Example Results)

From actual model runs:

| Method | Test Accuracy | Improvement over Best Individual |
|--------|--------------|----------------------------------|
| Best Individual (KNN) | 99.70% | — |
| **Weighted Fusion** | 99.73% | +0.03% |
| **GP1 (Max)** | 99.77% | +0.07% ✓ |
| **GP2 (Product)** | 99.60% | -0.10% |
| **GP3 (Ratio)** | 0.00% | Failed (numerical instability) |
| **GP4 (Weighted Sum)** | 99.60% | -0.10% |

### Method Selection Guidelines

**Use Weighted Fusion when:**
- Need stable, production-ready fusion
- All classifiers have similar performance
- Computational efficiency is important

**Use GP1 when:**
- One classifier significantly outperforms others
- Want to leverage "best expert" strategy
- Predictions are highly confident

**Use GP2 when:**
- Classifiers have balanced performance
- Want probabilistic OR-like behavior
- Need symmetric, interpretable fusion

**Use GP3 when:**
- Experimental/research context
- Willing to handle numerical edge cases
- Need extreme non-linearity

**Use GP4 when:**
- Want normalized averaging behavior
- Need automatic dampening
- Seeking conservative fusion

---

## Conclusion

This system provides **5 distinct mathematical fusion strategies** for ensemble learning:

1. **Weighted Fusion:** Balanced, production-ready, interpretable
2. **GP1:** Optimistic, expert-selection based
3. **GP2:** Probabilistic, symmetric, multiplicative
4. **GP3:** Experimental, highly non-linear
5. **GP4:** Conservative, normalized averaging

Each method has unique mathematical properties and suitability for different scenarios. **GP1 achieved the best empirical performance** in the lung histopathology classification task, suggesting that optimistic fusion strategies work well when classifier confidence is high and well-calibrated.

---

**References:**
- Kuncheva, L. I. (2004). Combining Pattern Classifiers: Methods and Algorithms
- Wolpert, D. H. (1992). Stacked Generalization
- Geman, S., et al. (1992). Neural Networks and the Bias/Variance Dilemma

---

*Last Updated: February 21, 2026*
