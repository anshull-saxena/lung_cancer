# Function Documentation - Lung Histopathology Classification

**Project:** Multi-CNN + Channel Attention + AGWO + Multi-Classifier Ensemble  
**Date:** November 13, 2025

---

## Table of Contents
1. [Multi-Head Channel Attention Functions](#multi-head-channel-attention-functions)
2. [Feature Extraction Functions](#feature-extraction-functions)
3. [Feature Selection Functions](#feature-selection-functions)
4. [Ensemble Fusion Functions](#ensemble-fusion-functions)
5. [Utility Functions](#utility-functions)
6. [Complete Function Call Flow](#complete-function-call-flow)

---

## Multi-Head Channel Attention Functions

### 1. `MultiHeadChannelAttention` (Class)

```python
class MultiHeadChannelAttention(Layer):
    def __init__(self, num_heads=4, reduction=16, **kwargs)
    def build(self, input_shape)
    def call(self, x)
```

**Purpose:**  
Implements a multi-head channel attention mechanism that learns to emphasize important feature channels while suppressing less useful ones.

**How it works:**
1. Takes feature maps from CNN backbone (shape: H × W × C)
2. Applies Global Average Pooling and Global Max Pooling
3. Passes pooled features through 2 dense layers to learn attention weights
4. Uses multiple "heads" (parallel attention pathways) for diverse patterns
5. Multiplies original features by learned attention weights

**Parameters:**
- `num_heads` (int): Number of parallel attention heads (default: 4, set to 8 in code)
- `reduction` (int): Channel reduction ratio for efficiency (default: 16)
- `**kwargs`: Additional Keras layer arguments

**Mathematical Formula:**
```
gap = GlobalAveragePooling(x)  # Shape: (batch, C)
gmp = GlobalMaxPooling(x)      # Shape: (batch, C)

# Learn attention through dense layers
gap_feat = Dense(num_heads × C/reduction, relu)(gap)
gmp_feat = Dense(num_heads × C/reduction, relu)(gmp)

gap_attn = Dense(num_heads × C)(gap_feat)
gmp_attn = Dense(num_heads × C)(gmp_feat)

# Combine and average across heads
combined = Reshape([num_heads, C])(gap_attn + gmp_attn)
attention = sigmoid(mean(combined, axis=1))  # Shape: (batch, C)

# Apply attention
output = x × attention
```

**Why Multiple Heads?**
- Each head can learn different attention patterns
- Similar to multi-head attention in Transformers
- Increases model capacity without excessive parameters

**Example:**
```python
# In the code, applied to each CNN backbone
x = DenseNet121(...)(input_image)  # Output: (7, 7, 1024)
x = MultiHeadChannelAttention(num_heads=8, reduction=16)(x)
# Now 1024 channels are re-weighted by learned importance
```

---

### 2. `multi_head_attention_block()`

```python
def multi_head_attention_block(x, reduction=16, name=None):
    NUM_ATTENTION_HEADS = 4
    return MultiHeadChannelAttention(num_heads=NUM_ATTENTION_HEADS, reduction=reduction, name=name)(x)
```

**Purpose:**  
Wrapper function for easier application of multi-head channel attention.

**Parameters:**
- `x` (Tensor): Input feature maps from CNN backbone
- `reduction` (int): Channel reduction ratio (default: 16)
- `name` (str): Layer name for identification

**Returns:**
- Attention-weighted feature maps

**Usage:**
```python
# Instead of calling the class directly, use this wrapper
x = multi_head_attention_block(x, reduction=16, name="attention_densenet")
```

---

## Feature Extraction Functions

### 3. `lane()`

```python
def lane(tensor, backbone="resnet", reduction=16):
```

**Purpose:**  
Creates a complete "processing lane" for one CNN backbone, including preprocessing, feature extraction, attention, and pooling.

**How it works:**
1. Applies backbone-specific preprocessing (normalization)
2. Loads pre-trained CNN backbone (frozen weights from ImageNet)
3. Applies multi-head channel attention
4. Applies Global Average Pooling to flatten features
5. Returns 1D feature vector

**Parameters:**
- `tensor` (Tensor): Input image tensor (224 × 224 × 3)
- `backbone` (str): CNN architecture to use
  - `"resnet"` → ResNet50
  - `"densenet"` → DenseNet121
  - `"efficientnet"` → EfficientNetB0
  - `"inception"` → InceptionV3
- `reduction` (int): Attention mechanism reduction ratio

**Returns:**
- 1D feature vector (size depends on backbone)
  - DenseNet121: 1024 features
  - ResNet50: 2048 features
  - EfficientNetB0: 1280 features
  - InceptionV3: 2048 features

**Architecture Flow:**
```
Input (224×224×3)
    ↓
Preprocessing (backbone-specific normalization)
    ↓
CNN Backbone (pre-trained on ImageNet)
    ↓ (Output: H × W × C feature maps)
Multi-Head Channel Attention
    ↓ (Re-weighted feature maps)
Global Average Pooling
    ↓ (Output: C-dimensional vector)
Return feature vector
```

**Example:**
```python
input_img = Input(shape=(224, 224, 3))
densenet_features = lane(input_img, "densenet", reduction=16)
resnet_features = lane(input_img, "resnet", reduction=16)
# densenet_features: 1024-dim, resnet_features: 2048-dim
```

**Why "Lane"?**
- Each backbone is a parallel processing "lane"
- All lanes process the same image independently
- Results are concatenated later for fusion

---

### 4. `extract_features()`

```python
def extract_features(generator):
```

**Purpose:**  
Batch-processes images through the multi-backbone feature extraction model to get deep features for all images in a dataset.

**How it works:**
1. Iterates through batches from the data generator
2. For each batch, passes images through the 4-backbone CNN model
3. Collects features and labels
4. Shows progress with timing estimates
5. Returns stacked numpy arrays

**Parameters:**
- `generator` (ImageDataGenerator): Keras data generator yielding (images, labels) batches

**Returns:**
- `X` (ndarray): Feature matrix, shape (n_samples, n_features)
  - n_features ≈ 1024 + 2048 + 1280 + 2048 = 6400
- `y` (ndarray): One-hot encoded labels, shape (n_samples, n_classes)

**Implementation Details:**
```python
def extract_features(generator):
    X, y = [], []
    steps = len(generator)  # Number of batches
    
    for i in range(steps):
        imgs, labels = next(generator)  # Get batch
        feats = feature_model.predict(imgs, verbose=0)  # Extract features
        X.append(feats)
        y.append(labels)
        
        # Progress tracking every 20 batches
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{steps}] processed...")
    
    return np.vstack(X), np.vstack(y)
```

**Performance:**
- **GPU (M2):** ~15-20 minutes for 800 images
- **CPU:** ~2-3 hours for 800 images
- **Progress Updates:** Every 20 batches with ETA

**Example Usage:**
```python
train_gen = train_datagen.flow_from_directory(...)
X_train, Y_train = extract_features(train_gen)
# X_train: (640, 6400), Y_train: (640, 3)
```

---

## Feature Selection Functions

### 5. `true_mrmr_feature_selection()`

```python
def true_mrmr_feature_selection(X, y_ohe, n_features=1000, sample_rows=1500, 
                                 var_thresh=0.01, redundancy_penalty=0.4):
```

**Purpose:**  
Performs minimum Redundancy Maximum Relevance (mRMR) feature selection to identify the most informative and least redundant features.

**Algorithm:**
1. **Variance Filtering:** Removes features with low variance (< threshold)
2. **Mutual Information:** Computes relevance of each feature to target labels
3. **Greedy Selection:** Iteratively selects features that:
   - Have high relevance (high MI with labels)
   - Have low redundancy (low correlation with already-selected features)

**Parameters:**
- `X` (ndarray): Feature matrix (n_samples, n_features)
- `y_ohe` (ndarray): One-hot encoded labels
- `n_features` (int): Number of features to select (default: 1000)
- `sample_rows` (int): Number of rows to sample for speed (default: 1500)
- `var_thresh` (float): Minimum variance threshold (default: 0.01)
- `redundancy_penalty` (float): Weight for redundancy term (default: 0.4)

**Returns:**
- `selected_indices` (list): Indices of selected features (global indices from original X)

**Mathematical Formula:**
```
For each unselected feature f:

    Relevance(f) = I(f; y)  # Mutual Information with labels
    
    Redundancy(f) = (1/|S|) × Σ_{s∈S} |ρ(f, s)|  # Avg correlation with selected
    
    Score(f) = Relevance(f) - λ × Redundancy(f)
    
    Select f with highest score
```

Where:
- `I(f; y)` = Mutual Information between feature and label
- `ρ(f, s)` = Pearson correlation between features
- `λ` = `redundancy_penalty` = 0.4
- `S` = Set of already-selected features

**Step-by-Step Process:**
```
1. Remove low-variance features (variance < 0.01)
   6400 features → ~5500 features

2. Sample rows for efficiency (2000 random samples)
   Speed up MI computation

3. Compute MI for all remaining features
   I(f₁; y), I(f₂; y), ..., I(f₅₅₀₀; y)

4. Greedy selection loop (700 iterations):
   a. Initialize: selected = []
   b. First feature: Select one with highest MI
   c. For each next feature:
      - Compute correlation with all selected features
      - Score = MI - 0.4 × avg_correlation
      - Select feature with highest score
   d. Repeat until 700 features selected

5. Return global indices of selected features
```

**Why mRMR?**
- **Better than pure MI:** Removes redundant features
- **Better than correlation:** Captures non-linear relationships
- **Balanced:** Finds features that are relevant AND diverse

**Example:**
```python
X_train = np.random.rand(800, 6400)  # 800 samples, 6400 features
y_train = np.eye(3)[np.random.randint(0, 3, 800)]  # 3-class one-hot

selected = true_mrmr_feature_selection(
    X_train, y_train, 
    n_features=700,
    sample_rows=2000,
    var_thresh=0.01,
    redundancy_penalty=0.4
)
# Returns: [234, 1567, 89, 3421, ...] (700 indices)

X_train_mrmr = X_train[:, selected]  # (800, 700)
```

**Time Complexity:**
- Variance filtering: O(d) where d = features
- MI computation: O(n × d) where n = samples
- Greedy selection: O(k × d × k) where k = selected features
- **Total:** ~30-60 seconds for 6400 features

---

### 6. `enhanced_agwo_feature_selection()`

```python
def enhanced_agwo_feature_selection(
    X_ranked, y_ohe, ranked_global_indices,
    n_wolves=20, n_iter=15, min_subset=500, max_subset=1500,
    row_sample=2500, knn_folds=3, rf_folds=2, rf_max_features=400,
    penalty_weight=0.015, patience=6, random_state=42, verbose=True
):
```

**Purpose:**  
Implements Enhanced Adaptive Grey Wolf Optimization for optimal feature subset selection using cross-validated classifier performance as fitness.

**Algorithm Overview:**  
Grey Wolf Optimization mimics the hunting behavior of grey wolves, where wolves search for prey (optimal feature subset) following a hierarchy:
- **Alpha (α):** Best solution (leader)
- **Beta (β):** Second-best (lieutenant)
- **Delta (δ):** Third-best (scout)
- **Omega (ω):** Rest of pack (followers)

**Parameters:**
- `X_ranked` (ndarray): Feature matrix with mRMR-selected features (n_samples, ~700)
- `y_ohe` (ndarray): One-hot encoded labels
- `ranked_global_indices` (list): Global indices of features in X_ranked
- `n_wolves` (int): Population size (default: 20)
- `n_iter` (int): Number of iterations (default: 15)
- `min_subset` (int): Minimum features to select (default: 500)
- `max_subset` (int): Maximum features to select (default: 1500)
- `row_sample` (int): Rows to sample for fitness evaluation (default: 2500)
- `knn_folds` (int): K-fold CV for KNN fitness (default: 3)
- `rf_folds` (int): K-fold CV for RF fitness (default: 2)
- `rf_max_features` (int): Max features for RF training (default: 400)
- `penalty_weight` (float): Feature count penalty (default: 0.015)
- `patience` (int): Early stopping patience (default: 6)
- `random_state` (int): Random seed
- `verbose` (bool): Print progress

**Returns:**
- `selected_global` (list): Indices of selected features (global indices)

**Step-by-Step Algorithm:**

```
1. INITIALIZATION (wolves represent feature subsets)
   • Create n_wolves random "position vectors" (each dimension = feature weight)
   • Position[i] ∈ ℝ^d where d = number of mRMR features (700)
   • Random values with sinusoidal perturbation for diversity

2. DECODING (convert position to feature subset)
   • Each wolf position is a continuous vector
   • Decode by selecting top-k features with highest weights
   • k grows logarithmically: 500 → 1500 across iterations
   
   decode(position, k):
       Add noise to position
       Return indices of k largest values

3. FITNESS EVALUATION (how good is this feature subset?)
   
   For each wolf's feature subset F:
   
   a. Sample 2500 rows (stratified by class)
   b. Extract features at indices F: X_subset = X[:, F]
   c. Scale with StandardScaler
   
   d. KNN Cross-Validation:
      • 5-fold stratified CV
      • KNN with k=5, distance-weighted
      • Compute mean accuracy: acc_knn
   
   e. Random Forest Cross-Validation:
      • 3-fold stratified CV
      • 150 trees, max_features='sqrt'
      • If |F| > 400, randomly sample 400 features
      • Compute mean accuracy: acc_rf
   
   f. Compute fitness:
      fitness = 0.7 × acc_knn + 0.3 × acc_rf - 0.012 × (|F| / max_subset)
      
      Components:
      • 0.7 × acc_knn: KNN performance (70% weight)
      • 0.3 × acc_rf: RF performance (30% weight)
      • -0.012 × size: Penalty for larger subsets
   
   g. Cache fitness to avoid recomputation

4. MAIN LOOP (iterate n_iter times)
   
   For iteration t = 1 to n_iter:
   
   a. Determine subset size for this iteration:
      k_budget = min_subset + (max_subset - min_subset) × log_growth(t)
      Example: 500 → 545 → 591 → ... → 1200
   
   b. Decode all wolves to feature subsets of size k_budget
   
   c. Evaluate fitness for each wolf's subset
   
   d. Identify hierarchy:
      • α = wolf with best fitness
      • β = wolf with 2nd-best fitness
      • δ = wolf with 3rd-best fitness
   
   e. Update global best if α is better
   
   f. Check early stopping:
      If no improvement for 'patience' iterations, stop
   
   g. Update exploration parameter:
      a = 2 × exp(-4 × t / n_iter)
      Decays from 2.0 → 0.0 (exploration → exploitation)
   
   h. Update each wolf position (except α, β, δ):
      
      For each omega wolf w:
      
      # Compute distances to leaders
      D_α = |C₁ · α - w|
      D_β = |C₂ · β - w|
      D_δ = |C₃ · δ - w|
      
      # Compute candidate positions
      X₁ = α - A₁ · D_α
      X₂ = β - A₂ · D_β
      X₃ = δ - A₃ · D_δ
      
      # Average the three influences
      w_new = (X₁ + X₂ + X₃) / 3
      
      Where:
      A₁, A₂, A₃ = 2a · rand() - a  (exploration coefficients)
      C₁, C₂, C₃ = 2 · rand()        (random vectors)
      rand() ∈ [0, 1]
   
   i. Apply mutation (15% probability):
      • Select random dimensions (0.5% of features)
      • Add Gaussian noise (σ = 0.3)
   
   j. Clip positions to valid range: [-2, 2]
   
   k. Diversity injection (if stagnating):
      If no improvement for (patience - 1) iterations:
      • Re-initialize worst 20% of wolves
      • Maintains population diversity

5. RETURN
   • Best feature subset found (α wolf's subset)
   • Map local indices → global indices
```

**Mathematical Equations:**

**Position Update:**
```
D⃗_α = |C⃗₁ · X⃗_α - X⃗|
D⃗_β = |C⃗₂ · X⃗_β - X⃗|
D⃗_δ = |C⃗₃ · X⃗_δ - X⃗|

X⃗₁ = X⃗_α - A⃗₁ · D⃗_α
X⃗₂ = X⃗_β - A⃗₂ · D⃗_β
X⃗₃ = X⃗_δ - A⃗₃ · D⃗_δ

X⃗(t+1) = (X⃗₁ + X⃗₂ + X⃗₃) / 3
```

**Coefficients:**
```
A⃗ = 2a⃗ · r⃗₁ - a⃗
C⃗ = 2 · r⃗₂

where:
a = 2 × exp(-4 × t/T)  # Decays from 2 to 0
r⃗₁, r⃗₂ ~ U(0,1)       # Random vectors
```

**Fitness Function:**
```
fitness(F) = 0.7 × CV_accuracy_KNN(F) 
           + 0.3 × CV_accuracy_RF(F)
           - 0.012 × (|F| / max_subset)
```

**Why This Works:**

1. **Multi-objective:** Balances accuracy and feature count
2. **CV-based fitness:** More robust than single train/test split
3. **Adaptive:** Subset size grows with iterations (coarse → fine search)
4. **Hierarchical:** Best solutions guide the search
5. **Diverse:** Mutation and diversity injection prevent premature convergence

**Example Usage:**
```python
# After mRMR selects 700 features
X_mrmr = X_train[:, mrmr_indices]  # (800, 700)

selected_features = enhanced_agwo_feature_selection(
    X_ranked=X_mrmr,
    y_ohe=Y_train,
    ranked_global_indices=mrmr_indices,
    n_wolves=25,
    n_iter=20,
    min_subset=500,
    max_subset=1200,
    verbose=True
)
# Output: [iteration progress...]
# Returns: [234, 1567, 89, ...] (1000-1200 global indices)

X_final = X_train[:, selected_features]  # (800, 1100)
```

**Time Complexity:**
- Per iteration: O(n_wolves × (KNN_CV + RF_CV))
- KNN CV: O(n × d × k × folds) where k = neighbors
- RF CV: O(n × d × log(n) × trees × folds)
- **Total:** ~10-15 minutes with GPU for 25 wolves × 20 iterations

**Output Example:**
```
[AGWO] iter 1/20 k=545 alpha=0.9606 best=0.9606 cache=25
[AGWO] iter 2/20 k=572 alpha=0.9606 best=0.9606 cache=50
[AGWO] iter 3/20 k=591 alpha=0.9620 best=0.9620 cache=75
...
[AGWO] iter 20/20 k=1200 alpha=0.9701 best=0.9701 cache=487
[AGWO] Complete: 1142 features, fitness=0.9701
```

---

### 7. Helper Function: `_subset_hash()`

```python
def _subset_hash(idxs):
    return hashlib.md5(np.asarray(idxs, dtype=np.int32).tobytes()).hexdigest()
```

**Purpose:**  
Creates a unique hash for a feature subset to enable fitness caching.

**Why Needed:**  
AGWO might evaluate the same feature subset multiple times. Caching avoids expensive re-computation of CV fitness.

**Returns:**
- MD5 hash string of feature indices

---

### 8. Helper Function: `eval_subset()`

```python
def eval_subset(local_idx):  # Nested inside enhanced_agwo_feature_selection()
```

**Purpose:**  
Evaluates the fitness of a specific feature subset using cross-validated KNN and RF classifiers.

**Process:**
1. Check fitness cache (return if already computed)
2. Extract features at specified indices
3. Scale features with StandardScaler
4. Run k-fold CV with KNN → compute mean accuracy
5. Run k-fold CV with RF → compute mean accuracy
6. Compute weighted fitness with size penalty
7. Cache and return fitness

---

### 9. Helper Function: `decode()`

```python
def decode(position, k):  # Nested inside enhanced_agwo_feature_selection()
```

**Purpose:**  
Converts a continuous position vector (wolf's location) to a discrete feature subset.

**How it works:**
1. Add small Gaussian noise for stability
2. Select top-k features with highest position values
3. Sort selected features by descending position value

**Returns:**
- Indices of k selected features

---

### 10. Helper Function: `subset_budget()`

```python
def subset_budget(iter_idx):  # Nested inside enhanced_agwo_feature_selection()
```

**Purpose:**  
Computes the target subset size for a given iteration using logarithmic growth.

**Formula:**
```
log_factor = log(iter + 2) / log(max_iter + 1)
k = min_subset + (max_subset - min_subset) × log_factor
```

**Growth Pattern:**
```
Iteration  1:  500 features
Iteration  5:  650 features
Iteration 10:  850 features
Iteration 15: 1050 features
Iteration 20: 1200 features
```

**Why Logarithmic?**
- Start small (fast exploration)
- Gradually increase (refined search)
- Smooth transition (avoids sudden jumps)

---

## Ensemble Fusion Functions

### 11. Weighted Probability Ensemble

```python
# Implemented inline in notebook cell
# Computes weighted ensemble using priority-based fusion
```

**Purpose:**  
Combines predictions from 5 classifiers using performance-weighted probability fusion.

**Algorithm:**

```python
# Step 1: Collect classifier names, accuracies, and probabilities
clf_names = ['KNN', 'SVM', 'RF', 'LR', 'XGB']
clf_accs = [knn_acc, svm_acc, rf_acc, lr_acc, xgb_acc]
proba_dict = {'KNN': knn_proba, 'SVM': svm_proba, ...}

# Step 2: Rank classifiers by accuracy (descending)
rank_idx = np.argsort(clf_accs)[::-1]
ranked_names = clf_names[rank_idx]  # e.g., ['XGB', 'SVM', 'RF', 'LR', 'KNN']
ranked_accs = clf_accs[rank_idx]    # e.g., [0.97, 0.96, 0.95, 0.93, 0.92]

# Step 3: Compute cumulative performance weights
T = [1.0]
for j in range(1, 5):
    T.append(T[-1] * ranked_accs[j-1])

# Example:
# T[0] = 1.0
# T[1] = 1.0 × 0.97 = 0.97
# T[2] = 0.97 × 0.96 = 0.9312
# T[3] = 0.9312 × 0.95 = 0.8846
# T[4] = 0.8846 × 0.93 = 0.8227

# Step 4: Normalize to get weight coefficients
epsilon = T / sum(T)

# Example:
# sum(T) = 4.71
# ε = [0.212, 0.206, 0.198, 0.188, 0.175]
# → Best classifier gets highest weight

# Step 5: Apply weighted fusion formula
for each class c:
    for each sample s:
        numerator = Σ(εᵢ × Pᵢ(s, c))
        product = Π(εᵢ × Pᵢ(s, c))
        P_ensemble(s, c) = numerator / (1 + product)

# Step 6: Final prediction
y_pred = argmax(P_ensemble, axis=1)
```

**Mathematical Formula:**

For each class $c$ and sample $s$:

$$
P_{\text{ensemble}}(s, c) = \frac{\sum_{i=1}^{5} \varepsilon_i \cdot P_i(s, c)}{1 + \prod_{i=1}^{5} \varepsilon_i \cdot P_i(s, c)}
$$

Where:
- $P_i(s, c)$ = Probability of class $c$ from classifier $i$ for sample $s$
- $\varepsilon_i$ = Weight for classifier $i$ (based on ranking)
- $\sum$ = Summation over all 5 classifiers
- $\prod$ = Product over all 5 classifiers

**Why This Formula?**

1. **Numerator (weighted sum):**
   - Linear combination of probabilities
   - Better classifiers have higher weights
   - Represents "consensus strength"

2. **Denominator (product term):**
   - Non-linear dampening factor
   - Reduces overconfidence when all classifiers are uncertain
   - If all $\varepsilon_i \cdot P_i(s,c)$ are small → denominator ≈ 1 (minimal dampening)
   - If all $\varepsilon_i \cdot P_i(s,c)$ are large → denominator > 1 (dampening)

3. **Combined effect:**
   - High consensus + high confidence → high probability
   - High consensus + low confidence → moderate probability
   - Low consensus → distributed probability

**Example Calculation:**

```python
# Suppose for a sample s and class c=0 (ACA):
ranked_names = ['XGB', 'SVM', 'RF', 'LR', 'KNN']
epsilon = [0.42, 0.40, 0.11, 0.05, 0.02]
probas = [0.95, 0.93, 0.89, 0.85, 0.82]  # P_i(s, ACA)

# Numerator
numerator = (0.42×0.95 + 0.40×0.93 + 0.11×0.89 + 0.05×0.85 + 0.02×0.82)
         = 0.399 + 0.372 + 0.098 + 0.043 + 0.016
         = 0.928

# Product term
product = (0.42×0.95) × (0.40×0.93) × (0.11×0.89) × (0.05×0.85) × (0.02×0.82)
        = 0.399 × 0.372 × 0.098 × 0.043 × 0.016
        = 0.000094

# Final probability
P_ensemble(s, ACA) = 0.928 / (1 + 0.000094) = 0.928 / 1.000094 ≈ 0.928
```

**Weight Distribution Example:**

```
If accuracies are: [0.97, 0.96, 0.95, 0.93, 0.92]

T values:
T₁ = 1.0
T₂ = 1.0 × 0.97 = 0.97
T₃ = 0.97 × 0.96 = 0.9312
T₄ = 0.9312 × 0.95 = 0.8846
T₅ = 0.8846 × 0.93 = 0.8227

Sum = 4.6085

Normalized weights (ε):
ε₁ = 1.0 / 4.6085 = 0.217 (21.7%) ← Best classifier
ε₂ = 0.97 / 4.6085 = 0.211 (21.1%)
ε₃ = 0.9312 / 4.6085 = 0.202 (20.2%)
ε₄ = 0.8846 / 4.6085 = 0.192 (19.2%)
ε₅ = 0.8227 / 4.6085 = 0.179 (17.9%) ← Worst classifier

→ Top 2 classifiers get 42.8% of total weight
```

---

## Utility Functions

### 12. Data Generator Setup

```python
train_datagen = ImageDataGenerator(
    validation_split=0.20,
    rotation_range=20,
    horizontal_flip=True
)

def make_gen(subset):
    return train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        subset=subset,
        seed=SEED,
        shuffle=True
    )
```

**Purpose:**  
Creates Keras data generators for loading and augmenting images during training.

**Features:**
- **validation_split=0.20:** 80% train, 20% validation
- **rotation_range=20:** Random rotation ±20 degrees
- **horizontal_flip=True:** Random horizontal flipping
- **No rescaling:** Preprocessing done per-backbone

**Usage:**
```python
train_gen = make_gen('training')   # Training subset
val_gen = make_gen('validation')   # Validation subset
```

---

## Complete Function Call Flow

### Training Pipeline

```
1. SETUP
   ├─ Load libraries
   ├─ Configure GPU (detect Metal/CUDA)
   ├─ Set random seeds (reproducibility)
   └─ Create data generators (train_gen, val_gen)

2. FEATURE EXTRACTION
   ├─ Build feature model:
   │  ├─ Input(224, 224, 3)
   │  ├─ lane(input, "densenet") → 1024 features
   │  ├─ lane(input, "resnet") → 2048 features
   │  ├─ lane(input, "efficientnet") → 1280 features
   │  ├─ lane(input, "inception") → 2048 features
   │  └─ Concatenate() → ~6400 features
   │
   ├─ extract_features(train_gen) → X_train, Y_train
   └─ extract_features(val_gen) → X_val, Y_val

3. FEATURE SELECTION (Stage 1: mRMR)
   ├─ true_mrmr_feature_selection(X_train, Y_train)
   │  ├─ Variance filtering
   │  ├─ Mutual Information computation
   │  └─ Greedy redundancy-penalized selection
   │
   └─ Output: ranked_features (700 indices)

4. FEATURE SELECTION (Stage 2: AGWO)
   ├─ Slice to mRMR features: X_train[:, ranked_features]
   │
   ├─ enhanced_agwo_feature_selection(X_train_mrmr, Y_train, ranked_features)
   │  ├─ Initialize 25 wolves (random positions)
   │  ├─ Main loop (20 iterations):
   │  │  ├─ decode(wolf_positions) → feature subsets
   │  │  ├─ eval_subset(subset):
   │  │  │  ├─ KNN CV (5-fold)
   │  │  │  ├─ RF CV (3-fold)
   │  │  │  └─ fitness = 0.7×knn + 0.3×rf - 0.012×size
   │  │  ├─ Identify α, β, δ wolves
   │  │  ├─ Update omega wolves (position formula)
   │  │  ├─ Apply mutation
   │  │  └─ Diversity injection if stagnating
   │  └─ Return best feature subset
   │
   └─ Output: selected_features (1000-1200 indices)

5. PREPARE FINAL FEATURES
   ├─ X_train_final = X_train[:, selected_features]
   ├─ X_val_final = X_val[:, selected_features]
   ├─ Combine: X_full = vstack([X_train_final, X_val_final])
   ├─ train_test_split(X_full, test_size=0.20)
   └─ Scale with RobustScaler

6. TRAIN CLASSIFIERS (5 in parallel)
   ├─ knn.fit(X_train_scaled, y_train)
   ├─ svm.fit(X_train_scaled, y_train)
   ├─ rf.fit(X_train_scaled, y_train)
   ├─ lr.fit(X_train_scaled, y_train)
   └─ xgb.fit(X_train_scaled, y_train)

7. MAKE PREDICTIONS
   ├─ knn_proba = knn.predict_proba(X_test_scaled)
   ├─ svm_proba = svm.predict_proba(X_test_scaled)
   ├─ rf_proba = rf.predict_proba(X_test_scaled)
   ├─ lr_proba = lr.predict_proba(X_test_scaled)
   └─ xgb_proba = xgb.predict_proba(X_test_scaled)

8. ENSEMBLE FUSION
   ├─ Rank classifiers by validation accuracy
   ├─ Compute cumulative weights (T values)
   ├─ Normalize to get ε weights
   ├─ Apply weighted fusion formula:
   │  └─ P(c) = Σ(εᵢ × Pᵢ(c)) / (1 + Π(εᵢ × Pᵢ(c)))
   └─ ensemble_pred = argmax(P(c))

9. EVALUATE & REPORT
   ├─ Individual accuracies (KNN, SVM, RF, LR, XGB)
   ├─ Ensemble accuracy
   ├─ Classification reports (per-class metrics)
   └─ Improvement over best individual
```

---

## Function Complexity Summary

| Function | Time Complexity | Space Complexity | Bottleneck |
|----------|----------------|------------------|------------|
| `MultiHeadChannelAttention.call()` | O(HWC + C²) | O(C) | Dense layers |
| `lane()` | O(HWC × depth) | O(HWC) | CNN forward pass |
| `extract_features()` | O(n × HWC × depth) | O(n × d) | GPU compute |
| `true_mrmr_feature_selection()` | O(d² × k) | O(d) | Correlation matrix |
| `enhanced_agwo_feature_selection()` | O(wolves × iters × CV) | O(wolves × d) | CV fitness |
| Ensemble fusion | O(n × classifiers × classes) | O(n × classes) | Probability computation |

Where:
- n = number of samples
- d = number of features
- H, W, C = height, width, channels
- depth = CNN layers
- k = selected features in mRMR

---

## Key Insights

### 1. Why Multi-Head Attention?
- **Single-head:** Limited expressiveness, one attention pattern
- **Multi-head:** Multiple parallel attention pathways
- **Benefit:** Captures diverse feature importance patterns (e.g., texture vs shape)

### 2. Why Two-Stage Feature Selection?
- **mRMR alone:** Fast but greedy (local optimum)
- **AGWO alone:** Slow on 6400 features
- **Combined:** mRMR reduces search space, AGWO optimizes within it

### 3. Why Multiple Classifiers?
- **Diversity:** Different algorithms have different biases
- **Complementarity:** Errors are uncorrelated
- **Robustness:** Ensemble reduces variance

### 4. Why Weighted Fusion (not voting)?
- **Voting:** Treats all classifiers equally (suboptimal)
- **Weighted:** Gives more power to better performers
- **Non-linear term:** Prevents overconfidence

### 5. Why Cross-Validation in AGWO?
- **Single split:** Unstable fitness (high variance)
- **CV:** More robust estimate of generalization
- **Cost:** 5× slower, but worth it for better features

---

## Common Issues & Debugging

### Issue 1: GPU Not Detected
```python
# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("No GPU detected")
    # Solution: Install tensorflow-metal (macOS) or tensorflow[and-cuda] (Linux)
```

### Issue 2: Out of Memory (OOM)
```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 24

# Or reduce number of wolves
n_wolves = 15  # Instead of 25
```

### Issue 3: AGWO Converges Too Early
```python
# Increase patience
patience = 10  # Instead of 8

# Increase diversity
mutation_prob = 0.20  # Instead of 0.15
```

### Issue 4: Poor Ensemble Performance
```python
# Check individual accuracies
print(f"KNN: {knn_acc}, SVM: {svm_acc}, ...")

# If all similar: ensemble won't help much
# If diverse: ensemble should improve
```

---

## Performance Optimization Tips

### 1. Feature Extraction
```python
# Use mixed precision (if supported)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Increase batch size (if GPU memory allows)
BATCH_SIZE = 32  # Higher = faster
```

### 2. mRMR
```python
# Reduce sample rows (faster but less accurate)
sample_rows = 1000  # Instead of 2000

# Increase variance threshold (fewer features)
var_thresh = 0.02  # Instead of 0.01
```

### 3. AGWO
```python
# Reduce CV folds (faster but less robust)
knn_folds = 3  # Instead of 5
rf_folds = 2  # Instead of 3

# Reduce iterations (faster but less optimal)
n_iter = 15  # Instead of 20
```

### 4. Classifier Training
```python
# Use n_jobs=-1 for parallelization
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)

# Reduce RF trees (faster but less accurate)
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1)
```

---

**Document Version:** 1.0  
**Last Updated:** November 13, 2025  
**Author:** Lung Cancer Classification Team
