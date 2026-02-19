"""
Global configuration for journal experiments.
Adaptive GA Deep Feature Selector for Lung Histopathological Image Classification.
"""
import os
import random
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "dataset", "lung_image_sets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "feature_cache")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]
NUM_CLASSES = 3

# ── Data split ────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# ── GA hyperparameters ────────────────────────────────────────────────────────
POP_SIZE = 40
N_GEN = 50
CX_PROB = 0.8
MUT_PROB = 0.1
INDPB = 0.05

# ── NSGA-II hyperparameters ──────────────────────────────────────────────────
NSGA_POP = 60
NSGA_GEN = 80

# ── Classifier hyperparameters ───────────────────────────────────────────────
KNN_K = 5
KNN_WEIGHTS = "distance"
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"
RF_N_ESTIMATORS = 300

# ── Cross-validation ─────────────────────────────────────────────────────────
CV_FOLDS = 3      # inner CV for GA fitness
K_FOLDS = 10      # outer k-fold for Table 4


def set_seed(seed=SEED):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
