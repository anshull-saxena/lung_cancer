#!/bin/bash
# =============================================================================
# Lung Cancer Experiment Runner
# Installs dependencies and runs all experiments on a cloud machine.
#
# Usage:
#   chmod +x execute.sh
#   ./execute.sh              # Run ALL experiments (Tables 2-16)
#   ./execute.sh 15 16        # Run only Tables 15 and 16 (new experiments)
#   ./execute.sh --new        # Run only the new experiments (Tables 15, 16)
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Lung Cancer Experiment Pipeline"
echo "  Working directory: $SCRIPT_DIR"
echo "=============================================="

# ── 1. Install dependencies ──────────────────────────────────────────────────
echo ""
echo "[1/3] Installing dependencies ..."

pip install --upgrade pip -q

# Core ML dependencies
pip install numpy scipy scikit-learn -q
pip install tensorflow -q
pip install xgboost -q

# DEAP for genetic algorithms
pip install deap -q

# Utilities
pip install Pillow -q

echo "  Dependencies installed."

# ── 2. Verify dataset ────────────────────────────────────────────────────────
echo ""
echo "[2/3] Verifying dataset ..."

DATASET_DIR="$SCRIPT_DIR/dataset/lung_image_sets"
if [ ! -d "$DATASET_DIR" ]; then
    echo "  ERROR: Dataset not found at $DATASET_DIR"
    echo "  Please ensure the LC25000 lung dataset is placed at:"
    echo "    $DATASET_DIR/lung_aca/"
    echo "    $DATASET_DIR/lung_n/"
    echo "    $DATASET_DIR/lung_scc/"
    exit 1
fi

for class_dir in lung_aca lung_n lung_scc; do
    count=$(ls "$DATASET_DIR/$class_dir" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $class_dir: $count images"
done

# ── 3. Run experiments ───────────────────────────────────────────────────────
echo ""
echo "[3/3] Running experiments ..."

cd "$SCRIPT_DIR/journal_experiments"

if [ "$1" = "--new" ]; then
    echo "  Running NEW experiments only (Tables 15, 16) ..."
    python run_all.py --tables 15 16
elif [ $# -gt 0 ]; then
    echo "  Running Tables: $@ ..."
    python run_all.py --tables "$@"
else
    echo "  Running ALL experiments ..."
    python run_all.py --all
fi

echo ""
echo "=============================================="
echo "  All experiments complete!"
echo "  Results saved to: $SCRIPT_DIR/journal_experiments/results/"
echo "=============================================="

# List generated result files
echo ""
echo "Generated files:"
ls -la "$SCRIPT_DIR/journal_experiments/results/"*.{json,csv,tex} 2>/dev/null || echo "  (no results yet)"
