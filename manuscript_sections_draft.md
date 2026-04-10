# Manuscript Sections Draft – Research Paper Content

**To:** Prof. Swati Hait
**From:** Anshul
**Date:** 21 February 2026
**Subject:** Draft Content for Three Manuscript Sections – NSGA-II Ensemble Framework for Lung Histopathology Classification

---

Dear Professor,

Please find below the draft content for the three requested manuscript sections: (1) Complete System Pipeline Flowchart Description, (2) NSGA-Based Feature Selection Implementation Details, and (3) Mathematical Functions and Evaluation Operators. All content is written in formal journal style and is intended for direct inclusion into the LaTeX manuscript.

---

## Section 1: Final Updated Flowchart – Complete System Pipeline

The following describes the full end-to-end architecture of the proposed lung histopathology classification system in logically ordered blocks. Each block specifies its inputs, outputs, and functional purpose, enabling direct conversion into a publication-ready flowchart diagram.

---

### Block 1: Dataset Acquisition

- **Input:** LC25000 lung histopathology benchmark dataset (Borkowski et al., 2019)
- **Output:** 15,000 RGB histopathological images across three classes: Lung Adenocarcinoma (lung\_aca, 5,000 images), Normal Lung Tissue (lung\_n, 5,000 images), and Lung Squamous Cell Carcinoma (lung\_scc, 5,000 images)
- **Purpose:** Provides the labeled image corpus for supervised classification of lung tissue subtypes.

### Block 2: Image Preprocessing

- **Input:** Raw histopathological images of variable internal representation
- **Output:** Normalized tensors of dimension 224 × 224 × 3 (float32)
- **Processing Steps:**
  - Spatial resizing to 224 × 224 pixels via bilinear interpolation
  - Backbone-specific pixel normalization (e.g., `keras.applications.densenet.preprocess_input` for DenseNet121, `keras.applications.resnet.preprocess_input` for ResNet50)
  - Data augmentation applied to the training partition: random rotation (±20°), horizontal flipping, and rescaling to [0, 1]
- **Purpose:** Standardizes input dimensions and intensity distributions to match the pre-training statistics of each backbone; augmentation regularizes training to reduce overfitting.

### Block 3: Stratified Data Partitioning

- **Input:** 15,000 preprocessed image–label pairs
- **Output:** Training set (70%, 10,500 samples), Validation set (10%, 1,500 samples), Test set (20%, 3,000 samples), with class proportions preserved across all partitions
- **Purpose:** Ensures unbiased evaluation by maintaining class balance; the training–validation split is used for feature selection fitness evaluation via cross-validation, while the test set is held out for final performance assessment.

### Block 4: Deep Feature Extraction via Pre-trained CNN Backbone with Channel Attention

- **Input:** Preprocessed image tensors (224 × 224 × 3)
- **Output:** 1-D feature vector per image (dimensionality depends on backbone: 1,024 for DenseNet121, 2,048 for ResNet50, 512 for VGG16, 1,280 for EfficientNetB0)
- **Architecture:**
  1. A pre-trained CNN backbone (ImageNet weights, `include_top=False`, all layers frozen) extracts convolutional feature maps **F** ∈ ℝ^(H′ × W′ × C).
  2. A Squeeze-and-Excitation (SE) channel attention module computes channel importance weights:

     **w** = σ(**W**₂ · ReLU(**W**₁ · GAP(**F**)))

     where GAP denotes Global Average Pooling, **W**₁ ∈ ℝ^(C/r × C) and **W**₂ ∈ ℝ^(C × C/r) are learnable projection matrices with reduction ratio r = 16, and σ is the sigmoid function. The SE module additionally incorporates Global Max Pooling, summing the two pathway outputs before the sigmoid gate.
  3. The attention-reweighted feature map **F**′ = **w** ⊙ **F** is passed through Global Average Pooling to produce a fixed-length 1-D vector **f** ∈ ℝ^C.
- **Purpose:** Leverages transfer learning from ImageNet to extract high-level morphological features; the SE attention mechanism suppresses irrelevant channels and amplifies diagnostically discriminative ones.
- **Caching:** Extracted features are serialized to `.npy` files to avoid redundant forward passes during iterative feature selection.

### Block 5: Feature Vector Construction

- **Input:** Per-image feature vectors from the backbone
- **Output:** Feature matrix **X** ∈ ℝ^(N × D), where N is the number of samples and D is the feature dimensionality (e.g., D = 1,024 for DenseNet121)
- **Purpose:** Assembles the complete feature representation for the entire dataset into a structured matrix amenable to evolutionary feature selection.

### Block 6: Multi-Objective Feature Selection via NSGA-II

- **Input:** Feature matrix **X** ∈ ℝ^(N × D) and corresponding label vector **y** ∈ {0, 1, 2}^N
- **Output:** Binary selection mask **s** ∈ {0, 1}^D defining the optimal feature subset; reduced feature matrix **X**\_sel ∈ ℝ^(N × d), where d ≪ D
- **Processing:**
  1. Initialize a population of P = 60 binary chromosomes, each of length D.
  2. Evaluate two competing objectives for each individual: (i) maximize 3-fold cross-validated classification accuracy using KNN (k = 5, distance-weighted), and (ii) minimize the number of selected features.
  3. Apply fast non-dominated sorting to assign Pareto ranks.
  4. Compute crowding distance within each front to preserve solution diversity.
  5. Generate offspring via binary tournament selection, uniform crossover (probability 0.8), and bit-flip mutation (per-gene probability 0.05, applied with probability 0.1).
  6. Combine parent and offspring populations, re-sort, and select the next generation front-by-front using crowding distance as the tiebreaker.
  7. Repeat for G = 80 generations.
  8. From the final Pareto front, apply the knee-point heuristic: filter solutions with feature count ≤ median, then select the one with highest accuracy.
- **Purpose:** Simultaneously optimizes classification performance and model parsimony without collapsing the two objectives into a single weighted scalar; produces an explicit accuracy–complexity trade-off surface.

### Block 7: Selected Feature Subset

- **Input:** Binary mask **s** from NSGA-II and original feature matrix **X**
- **Output:** Reduced feature matrix **X**\_sel containing only the d selected feature columns
- **Purpose:** Eliminates redundant and noisy features, reducing computational cost and mitigating overfitting in downstream classifiers.

### Block 8: Ensemble Classification

- **Input:** Reduced feature matrix **X**\_sel ∈ ℝ^(N × d) (training partition for fitting, test partition for inference)
- **Output:** Predicted class labels ŷ ∈ {0, 1, 2}^(N\_test) for each test sample
- **Classifiers:**
  1. K-Nearest Neighbors (k = 5, distance-weighted, Euclidean metric)
  2. Support Vector Machine (RBF kernel, C = 1.0, γ = scale, probability calibration enabled)
  3. Random Forest (300 estimators, no maximum depth constraint)
- **Fusion Strategy:** Hard majority voting—the final prediction for each sample is the class label receiving the plurality of votes across the three classifiers. Ties are resolved by `scipy.stats.mode`.
- **Purpose:** Combines classifiers with diverse inductive biases (instance-based, kernel-based, tree-based) to reduce individual classifier variance and improve generalization robustness.

### Block 9: Performance Evaluation

- **Input:** Predicted labels ŷ and ground-truth labels **y** for the held-out test set
- **Output:** Accuracy, Precision (macro-averaged), Recall/Sensitivity (macro-averaged), F1-Score (macro-averaged), Specificity (macro-averaged), per-class metrics, and the full confusion matrix
- **Purpose:** Provides a comprehensive multi-metric assessment of classification performance, with specificity computed from the confusion matrix to quantify the classifier's ability to correctly identify negative cases for each class.

---

## Section 2: NSGA-Based Feature Selection – Implementation Details

### 2.1 Problem Formulation

Feature selection in high-dimensional deep feature spaces can be formulated as a combinatorial optimization problem. Given a feature matrix **X** ∈ ℝ^(N × D) extracted by the CNN backbone, the objective is to identify a binary selection vector **s** = (s₁, s₂, …, s_D) ∈ {0, 1}^D that simultaneously satisfies two conflicting goals: maximizing classification performance and minimizing the number of selected features.

Classical Genetic Algorithms (GAs) address this as a single-objective problem by combining accuracy and a sparsity penalty into a scalar fitness function:

$$f_{GA}(\mathbf{s}) = \text{Acc}(\mathbf{s}) - \lambda \cdot \frac{\|\mathbf{s}\|_0}{D}$$

where λ is a manually tuned penalty coefficient (λ = 0.001 in the baseline implementation). This formulation suffers from two fundamental limitations. First, the choice of λ critically determines the accuracy–sparsity trade-off, yet no principled method exists for setting this hyperparameter *a priori*. Second, the scalar aggregation destroys the Pareto structure of the problem, collapsing the space of optimal trade-offs into a single point and discarding potentially valuable solutions.

NSGA-II (Non-dominated Sorting Genetic Algorithm II) overcomes these limitations by treating feature selection as a true bi-objective optimization problem, maintaining a population of Pareto-optimal solutions that represent the full spectrum of accuracy–complexity trade-offs.

### 2.2 Chromosome Representation

Each individual in the NSGA-II population is encoded as a binary chromosome **s** = (s₁, s₂, …, s_D) of length D, where D equals the dimensionality of the extracted feature space. Each gene s_i ∈ {0, 1} indicates whether feature i is included (s_i = 1) or excluded (s_i = 0) from the selected subset. The search space is therefore {0, 1}^D, with 2^D possible feature subsets. A minimum cardinality constraint enforces ‖**s**‖₀ ≥ 2 to ensure that at least two features are available for classification; individuals violating this constraint receive a penalized fitness of (0.0, ‖**s**‖₀).

### 2.3 Population Initialization

The initial population of P = 60 individuals is generated by independently sampling each gene from a Bernoulli distribution with probability 0.5. This uniform initialization ensures unbiased exploration of the feature space without presupposing any feature subset structure. The random seed is fixed (seed = 42) for reproducibility.

### 2.4 Objective Functions

Two objectives are evaluated for each individual **s**:

**Objective 1 – Maximize Classification Accuracy:**

$$f_1(\mathbf{s}) = \frac{1}{K} \sum_{k=1}^{K} \text{Acc}(h(\mathbf{X}_{sel}^{train,k}), \mathbf{X}_{sel}^{val,k}, \mathbf{y}^{val,k})$$

where K = 3 is the number of cross-validation folds, **X**\_sel^{train,k} and **X**\_sel^{val,k} denote the training and validation partitions of fold k restricted to the selected features, h(·) denotes the KNN classifier (k = 5, distance-weighted) trained on the training partition, and

$$\text{Acc}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}[\hat{y}_j = y_j]$$

is the classification accuracy, with 𝟙[·] denoting the indicator function.

**Objective 2 – Minimize Number of Selected Features:**

$$f_2(\mathbf{s}) = \|\mathbf{s}\|_0 = \sum_{i=1}^{D} s_i$$

This objective directly penalizes model complexity by counting the number of active features.

The bi-objective optimization problem is thus:

$$\max f_1(\mathbf{s}), \quad \min f_2(\mathbf{s}), \quad \text{subject to } \mathbf{s} \in \{0,1\}^D, \; \|\mathbf{s}\|_0 \geq 2$$

### 2.5 Dominance Relation and Non-Dominated Sorting

A solution **s**^(a) is said to *dominate* another solution **s**^(b), denoted **s**^(a) ≻ **s**^(b), if and only if:

1. f₁(**s**^(a)) ≥ f₁(**s**^(b)) and f₂(**s**^(a)) ≤ f₂(**s**^(b)) — at least as good in both objectives, **and**
2. f₁(**s**^(a)) > f₁(**s**^(b)) or f₂(**s**^(a)) < f₂(**s**^(b)) — strictly better in at least one objective.

**Fast Non-Dominated Sorting** partitions the population into successive Pareto fronts F₁, F₂, …, F_L:

- F₁ (rank 1) contains all non-dominated individuals (the Pareto front).
- F_l (rank l) contains individuals dominated only by members of F₁ ∪ F₂ ∪ … ∪ F_{l−1}.

The algorithm operates in O(MN²) time, where M = 2 is the number of objectives and N = 2P is the combined parent–offspring population size. For each individual p, the algorithm computes:

- n_p: the number of individuals that dominate p
- S_p: the set of individuals that p dominates

Individuals with n_p = 0 constitute F₁. For each p ∈ F₁, n_q is decremented for all q ∈ S_p; individuals whose domination count reaches zero form F₂. This process continues until all individuals are assigned to a front.

### 2.6 Crowding Distance Calculation

Within each front F_l, crowding distance quantifies the density of solutions surrounding a given individual, serving as a diversity preservation mechanism. For each objective m ∈ {1, 2}, individuals in F_l are sorted by their m-th objective value. The crowding distance for individual i is computed as:

$$d_i = \sum_{m=1}^{M} \frac{f_m^{(i+1)} - f_m^{(i-1)}}{f_m^{max} - f_m^{min}}$$

where f_m^{(i+1)} and f_m^{(i−1)} are the m-th objective values of the neighboring individuals in the sorted order, and f_m^{max}, f_m^{min} are the maximum and minimum values of objective m within the front. Boundary individuals (those with the extreme objective values) are assigned an infinite crowding distance, ensuring their preservation across generations.

### 2.7 Selection Strategy

Parent selection employs **binary tournament selection** with the following precedence rules:

1. The individual with the lower Pareto rank (better front) is preferred.
2. If both individuals share the same rank, the one with the higher crowding distance is preferred.

This selection pressure drives the population toward the Pareto front while maintaining diversity along it. Tournament size k = 2 provides moderate selection pressure, balancing convergence speed and population diversity.

### 2.8 Crossover Operator

**Uniform crossover** is applied with probability p_c = 0.8. Given two parent chromosomes **s**^(a) and **s**^(b), each gene position i is independently assigned to either parent with equal probability:

$$s_i^{child} = \begin{cases} s_i^{(a)} & \text{if } r_i < 0.5 \\ s_i^{(b)} & \text{otherwise} \end{cases} \quad \text{for } i = 1, 2, \ldots, D$$

where r_i ~ Uniform(0, 1). This operator is preferred over one-point or two-point crossover for binary feature selection because it treats each feature independently, avoiding positional bias that would otherwise couple adjacent features in the chromosome. When group structure is specified (e.g., quartile-based feature groups), uniform crossover is applied independently within each group segment, preserving intra-group feature co-selection patterns.

### 2.9 Mutation Operator

**Bit-flip mutation** is applied with probability p_m = 0.1. When triggered, each gene s_i is independently flipped with per-gene probability p_{indpb} = 0.05:

$$s_i^{mutated} = \begin{cases} 1 - s_i & \text{if } u_i < p_{indpb} \\ s_i & \text{otherwise} \end{cases} \quad \text{for } i = 1, 2, \ldots, D$$

where u_i ~ Uniform(0, 1). The two-level stochastic mechanism (individual-level p_m and gene-level p_{indpb}) ensures that mutation events are infrequent but, when they occur, affect a moderate number of genes, enabling both local refinement and occasional exploratory jumps.

### 2.10 Environmental Selection and Elitism

The complete generational cycle proceeds as follows:

1. Generate P offspring from the current population via tournament selection, crossover, and mutation.
2. Merge parent and offspring populations into a combined pool R_t of size 2P.
3. Apply fast non-dominated sorting to R_t, yielding fronts F₁, F₂, ….
4. Fill the next generation P_{t+1} front by front: add all individuals from F₁, then F₂, and so on until adding the next complete front would exceed P.
5. For the last partial front F_l, compute crowding distances and sort individuals in descending order of crowding distance; select the top remaining slots.

This (μ + λ) elitist strategy guarantees that the best non-dominated solutions are never lost, as the entire Pareto front from the previous generation competes directly with offspring for survival.

### 2.11 Termination Criteria

The algorithm terminates after a fixed number of G = 80 generations. At termination, the final Pareto front F₁ contains the set of all non-dominated trade-off solutions.

### 2.12 Best Compromise Solution Selection

From the final Pareto front, the best compromise solution is selected via a median-based knee-point heuristic:

1. Compute the median feature count among all Pareto-optimal solutions: d_med = median({f₂(**s**) : **s** ∈ F₁})
2. Filter candidates: C = {**s** ∈ F₁ : f₂(**s**) ≤ d_med}
3. Select the solution with maximum accuracy: **s*** = argmax_{**s** ∈ C} f₁(**s**)

This heuristic balances accuracy and parsimony by restricting the selection to the sparser half of the Pareto front, then choosing the most accurate solution within that region.

### 2.13 Advantages of NSGA-II over Traditional GA

**Explicit multi-objective handling.** Traditional GA collapses multiple objectives into a single scalar fitness via penalty weighting. NSGA-II maintains the true Pareto structure, preserving a diverse set of optimal trade-off solutions. This eliminates the need to tune penalty coefficients and provides practitioners with the complete accuracy–complexity frontier.

**Superior diversity preservation.** The crowding distance mechanism actively prevents population convergence to a narrow region of the objective space. In contrast, traditional GA with tournament selection on scalar fitness tends toward premature convergence, particularly when the fitness landscape contains deceptive local optima. The combination of non-dominated sorting and crowding distance ensures that both highly accurate solutions and highly sparse solutions persist in the population.

**Avoidance of premature convergence.** Traditional GA populations often converge prematurely when selection pressure overwhelms genetic diversity. NSGA-II mitigates this through two mechanisms: (i) the Pareto ranking distributes selection pressure across multiple fronts rather than concentrating it on a single best individual, and (ii) the crowding distance tiebreaker actively favors individuals in less crowded regions, maintaining phenotypic diversity along the Pareto front.

**Robust feature subset diversity.** Because NSGA-II maintains a population of non-dominated solutions spanning different accuracy–complexity trade-offs, the algorithm implicitly explores diverse feature subsets throughout the evolutionary process. Solutions with high accuracy but many features coexist with solutions that are highly sparse but slightly less accurate. This population-level diversity prevents the search from becoming trapped in local optima corresponding to a single feature subset structure.

---

## Section 3: Mathematical Functions and Evaluation Operators

### 3.1 Feature Selection Objective Functions

Let **s** = (s₁, s₂, …, s_D) ∈ {0, 1}^D denote a binary feature selection vector, where D is the total number of extracted features. Define the selected feature index set as I(**s**) = {i : s_i = 1} and the reduced feature matrix as **X**\_sel = **X**[:, I(**s**)] ∈ ℝ^(N × d), where d = |I(**s**)| = ‖**s**‖₀.

**Objective 1: Maximize Classification Accuracy**

$$f_1(\mathbf{s}) = \frac{1}{K} \sum_{k=1}^{K} \text{Acc}\big(h(\mathbf{X}_{sel}^{train,k}),\; \mathbf{X}_{sel}^{val,k},\; \mathbf{y}^{val,k}\big)$$

where K = 3 is the number of cross-validation folds, **X**\_sel^{train,k} and **X**\_sel^{val,k} denote the training and validation partitions of fold k restricted to the selected features, h(·) denotes the KNN classifier trained on the training partition, and

$$\text{Acc}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}[\hat{y}_j = y_j]$$

is the classification accuracy, with 𝟙[·] denoting the indicator function.

**Objective 2: Minimize Number of Selected Features**

$$f_2(\mathbf{s}) = \|\mathbf{s}\|_0 = \sum_{i=1}^{D} s_i$$

**Objective 3 (Implicit via Objective 1): Maximize Sensitivity**

Sensitivity (recall) is implicitly maximized through Objective 1, as accuracy on a balanced dataset (equal class priors) is equivalent to macro-averaged recall. For the LC25000 dataset with equal class sizes (5,000 per class), maximizing accuracy directly maximizes the average per-class detection rate:

$$\text{Sensitivity}_{macro} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}$$

where C = 3 is the number of classes, TP_c is the number of true positives for class c, and FN_c is the number of false negatives for class c. On a perfectly balanced dataset, Acc = Sensitivity\_macro.

### 3.2 Evaluation Metrics

Let **y** ∈ {1, 2, …, C}^N denote the ground-truth labels and ŷ ∈ {1, 2, …, C}^N denote the predicted labels for N test samples across C = 3 classes.

**Confusion Matrix:**

The confusion matrix **M** ∈ ℤ^(C × C) is defined as:

$$M_{ij} = \big|\{n : y_n = i \;\wedge\; \hat{y}_n = j\}\big|$$

where M_{ij} counts the number of samples with true class i predicted as class j. The diagonal entries M_{ii} represent correct classifications.

For each class c, the following quantities are derived from **M**:

$$TP_c = M_{cc}$$

$$FP_c = \sum_{\substack{i=1 \\ i \neq c}}^{C} M_{ic} \quad \text{(column sum minus diagonal)}$$

$$FN_c = \sum_{\substack{j=1 \\ j \neq c}}^{C} M_{cj} \quad \text{(row sum minus diagonal)}$$

$$TN_c = \sum_{i=1}^{C} \sum_{j=1}^{C} M_{ij} - TP_c - FP_c - FN_c$$

**Accuracy:**

$$\text{Accuracy} = \frac{\sum_{c=1}^{C} TP_c}{N} = \frac{\text{Tr}(\mathbf{M})}{N}$$

**Precision (Macro-Averaged):**

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

$$\text{Precision}_{macro} = \frac{1}{C} \sum_{c=1}^{C} \text{Precision}_c$$

**Recall / Sensitivity (Macro-Averaged):**

$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$\text{Recall}_{macro} = \frac{1}{C} \sum_{c=1}^{C} \text{Recall}_c$$

**F1-Score (Macro-Averaged):**

$$F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$F1_{macro} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

**Specificity (Macro-Averaged):**

$$\text{Specificity}_c = \frac{TN_c}{TN_c + FP_c}$$

$$\text{Specificity}_{macro} = \frac{1}{C} \sum_{c=1}^{C} \text{Specificity}_c$$

Specificity quantifies the true negative rate—the proportion of non-class-c samples correctly identified as not belonging to class c. In a multi-class setting, this metric is particularly informative for evaluating whether the classifier avoids false alarms for each tissue subtype.

### 3.3 Ensemble Classification Strategy

**Majority Voting Fusion:**

Given an ensemble of L = 3 classifiers {h₁, h₂, h₃} (KNN, SVM, Random Forest), each classifier independently predicts a class label for test sample **x**:

$$\hat{y}_l = h_l(\mathbf{x}), \quad l = 1, 2, 3$$

The ensemble prediction is determined by hard majority voting:

$$\hat{y}_{ensemble}(\mathbf{x}) = \arg\max_{c \in \{1, \ldots, C\}} \sum_{l=1}^{L} \mathbb{1}[\hat{y}_l(\mathbf{x}) = c]$$

where 𝟙[·] is the indicator function. The predicted class is the one receiving the greatest number of votes across the L classifiers. In the case of a three-classifier ensemble with C = 3 classes, a majority requires at least 2 out of 3 agreeing votes. Ties (each classifier predicting a different class) are resolved by selecting the statistical mode.

**Rationale for Majority Voting:**

The three base classifiers—KNN (instance-based), SVM (kernel-based), and Random Forest (tree-based)—possess fundamentally different inductive biases and decision boundaries. KNN partitions the feature space based on local neighborhood density, SVM constructs a maximum-margin hyperplane in a kernel-induced Hilbert space, and Random Forest aggregates axis-aligned decision trees trained on bootstrap samples. This architectural diversity satisfies the necessary condition for ensemble gain: individual classifiers make errors on different subsets of the input space. Majority voting exploits this complementarity by correcting individual misclassifications when at least two of the three classifiers agree on the correct label.

---

Please review and share any feedback or revisions needed.

Best regards,
Anshul
