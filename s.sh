#!/usr/bin/env bash
# s.sh — GPU diagnosis and auto-fix helper for this project
# - Checks NVIDIA GPU presence and utilization (nvidia-smi) when available
# - Validates Python/PIP alignment and TensorFlow GPU readiness
# - Compares requirements.txt with installed packages and reconciles mismatches
# - Applies fixes automatically (override with SKIP_INSTALL=1)
#
# Usage:
#   bash s.sh                # run with auto-fix
#   SKIP_INSTALL=1 bash s.sh # run checks only, no changes
#   VERBOSE=1 bash s.sh      # more logging
#
# Notes:
# - On macOS (Apple Silicon/Intel): installs/validates tensorflow-macos and tensorflow-metal
# - On Linux (NVIDIA): installs/validates tensorflow[and-cuda]
# - Does NOT install OS-level NVIDIA drivers or CUDA toolkit; prints guidance instead

set -euo pipefail
IFS=$'\n\t'

VERBOSE=${VERBOSE:-0}
SKIP_INSTALL=${SKIP_INSTALL:-0}

log()  { printf "%s\n" "$*"; }
info() { printf "[INFO] %s\n" "$*"; }
warn() { printf "\033[33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[31m[ERROR]\033[0m %s\n" "$*"; }
ok()   { printf "\033[32m[OK]\033[0m %s\n" "$*"; }
die()  { err "$*"; exit 1; }

vecho() { if [ "${VERBOSE}" = "1" ]; then printf "[VERBOSE] %s\n" "$*"; fi; }

command_exists() { command -v "$1" >/dev/null 2>&1; }
script_dir() { cd "$(dirname "$0")" && pwd; }
repo_root="$(script_dir)"

OS="$(uname -s)"
ARCH="$(uname -m)"
info "OS: ${OS} | Arch: ${ARCH} | Shell: ${SHELL}"

# Prefer python3, fallback to python
PYTHON_BIN=${PYTHON_BIN:-}
if [ -z "${PYTHON_BIN}" ]; then
  if command_exists python3; then PYTHON_BIN="python3";
  elif command_exists python; then PYTHON_BIN="python";
  else die "Python not found. Please install Python 3.10/3.11 and retry."; fi
fi

# Always use matching pip via python -m pip
PIP_BIN=("${PYTHON_BIN}" -m pip)

python_ver="$(${PYTHON_BIN} -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
info "Python: ${PYTHON_BIN} (${python_ver})"

# Check if pip maps to same interpreter
pip_py_ver="$(${PYTHON_BIN} -m pip -V 2>/dev/null || true)"
vecho "pip: ${pip_py_ver}"

# Detect NVIDIA GPU via nvidia-smi (Linux/WSL typically)
have_nvidia_smi=0
if command_exists nvidia-smi; then
  have_nvidia_smi=1
  ok "nvidia-smi found"
else
  warn "nvidia-smi not found (expected only on NVIDIA Linux/WSL setups)."
fi

# Gather GPU info if available
GPU_PRESENT=0
GPU_UTIL=0
if [ ${have_nvidia_smi} -eq 1 ]; then
  # List GPUs
  if nvidia-smi -L >/dev/null 2>&1; then
    GPU_PRESENT=1
    nvidia-smi -L || true
    # Query utilization and running processes
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 0)
    GPU_UTIL=${util:-0}
    info "GPU utilization (first GPU): ${GPU_UTIL}%"
  else
    warn "nvidia-smi present but cannot list GPUs. Driver may be missing or restricted."
  fi
fi

# macOS specific GPU note
if [ "${OS}" = "Darwin" ]; then
  if [ "${ARCH}" = "arm64" ]; then
    info "Apple Silicon detected. TensorFlow uses Metal (tensorflow-macos + tensorflow-metal)."
  else
    info "macOS Intel detected. GPU acceleration typically limited unless using eGPU; Metal path still applies."
  fi
fi

# Inspect TensorFlow install and GPU visibility
TF_VERSION=""
TF_BUILT_WITH_CUDA=""
TF_GPUS=""
TF_IMPORT_OK=0
${PYTHON_BIN} - <<'PY' || true
try:
    import tensorflow as tf
    from tensorflow.python.platform import build_info as tf_build
    print("TF_OK=1")
    print("TF_VERSION=" + tf.__version__)
    try:
        print("TF_BUILT_WITH_CUDA=" + str(tf.test.is_built_with_cuda()))
    except Exception:
        print("TF_BUILT_WITH_CUDA=unknown")
    gpus = tf.config.list_physical_devices('GPU')
    print("TF_GPUS=" + str(len(gpus)))
except Exception as e:
    print("TF_OK=0")
    print("TF_ERR=" + str(e))
PY
) | while IFS= read -r line; do
  case "$line" in
    TF_OK=1) TF_IMPORT_OK=1 ;;
    TF_VERSION=*) TF_VERSION="${line#TF_VERSION=}" ;;
    TF_BUILT_WITH_CUDA=*) TF_BUILT_WITH_CUDA="${line#TF_BUILT_WITH_CUDA=}" ;;
    TF_GPUS=*) TF_GPUS="${line#TF_GPUS=}" ;;
    TF_ERR=*) warn "TensorFlow import error: ${line#TF_ERR=}" ;;
  esac
  vecho "$line"
done

if [ "${TF_IMPORT_OK}" = "1" ]; then
  ok "TensorFlow detected: ${TF_VERSION} (built_with_cuda=${TF_BUILT_WITH_CUDA}, visible_gpus=${TF_GPUS})"
else
  warn "TensorFlow not installed/usable in current interpreter."
fi

# Decide desired TF stack based on OS
DESIRED_TF_PKG=""
if [ "${OS}" = "Linux" ]; then
  if [ ${GPU_PRESENT} -eq 1 ]; then
    DESIRED_TF_PKG="tensorflow[and-cuda]==2.15.0"
  else
    DESIRED_TF_PKG="tensorflow==2.15.0"
  fi
elif [ "${OS}" = "Darwin" ]; then
  # Prefer macOS-native builds
  DESIRED_TF_PKG="tensorflow-macos==2.15.0"
else
  DESIRED_TF_PKG="tensorflow==2.15.0"
fi

# Determine if TF (and plugin on macOS) needs install/adjust
NEED_TF_INSTALL=0
if [ "${TF_IMPORT_OK}" != "1" ]; then
  NEED_TF_INSTALL=1
else
  # If Linux+NVIDIA but not GPU build, prefer switching
  if [ "${OS}" = "Linux" ] && [ ${GPU_PRESENT} -eq 1 ]; then
    if [ "${TF_BUILT_WITH_CUDA}" != "True" ] && [ "${TF_BUILT_WITH_CUDA}" != "true" ]; then
      warn "TensorFlow is not CUDA-enabled on an NVIDIA system."
      NEED_TF_INSTALL=1
    fi
  fi
  # On macOS, ensure tensorflow-metal if Apple Silicon or macOS
  if [ "${OS}" = "Darwin" ]; then
    HAS_METAL=$(${PYTHON_BIN} -c 'import importlib.util; import sys; print(1 if importlib.util.find_spec("tensorflow" ) else 0); print(1 if importlib.util.find_spec("tensorflow-metal") else 0)' 2>/dev/null | tail -n 1 || echo 0)
    if [ "${HAS_METAL}" != "1" ]; then
      warn "tensorflow-metal not detected; will install for GPU via Metal."
      NEED_TF_INSTALL=1
    fi
  fi
fi

# Build list of fixes to apply
declare -a INSTALL_QUEUE
if [ ${NEED_TF_INSTALL} -eq 1 ] && [ "${SKIP_INSTALL}" != "1" ]; then
  info "Queueing install: ${DESIRED_TF_PKG}"
  INSTALL_QUEUE+=("${DESIRED_TF_PKG}")
  if [ "${OS}" = "Darwin" ]; then
    INSTALL_QUEUE+=("tensorflow-metal==1.1.0")
  fi
fi

# Check requirements.txt mismatches
REQ_FILE="${repo_root}/requirements.txt"
if [ -f "${REQ_FILE}" ]; then
  info "Checking requirements.txt for mismatches..."
  # Simple strategy: verify packages are importable; if not, reinstall constrained by requirements.txt
  if [ "${SKIP_INSTALL}" != "1" ]; then
    info "Installing/aligning requirements via: ${PYTHON_BIN} -m pip install -r requirements.txt"
    "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}" 1>/dev/null
    ok "requirements.txt installed"
  else
    warn "SKIP_INSTALL=1 set; not installing requirements.txt"
  fi
else
  warn "requirements.txt not found at repo root; skipping requirements alignment"
fi

# Apply queued installs (TensorFlow stack) after requirements to avoid being overwritten
if [ ${#INSTALL_QUEUE[@]} -gt 0 ]; then
  if [ "${SKIP_INSTALL}" != "1" ]; then
    info "Installing queued packages: ${INSTALL_QUEUE[*]}"
    "${PYTHON_BIN}" -m pip install --upgrade --no-cache-dir "${INSTALL_QUEUE[@]}"
    ok "TensorFlow stack installation finished"
  else
    warn "Skipped installing queued packages due to SKIP_INSTALL=1"
  fi
fi

# Post-install validation: import tf and run tiny GPU test
POST_TF_OK=0
POST_TF_GPUS=0
${PYTHON_BIN} - <<'PY' || true
import os, sys
try:
    import tensorflow as tf
    print("OK=1")
    g = tf.config.list_physical_devices('GPU')
    print("NGPU=" + str(len(g)))
    # Quick matmul to exercise GPU if present
    if g:
        with tf.device('/GPU:0'):
            a = tf.random.normal((2048, 2048))
            b = tf.random.normal((2048, 2048))
            c = tf.linalg.matmul(a, b)
        print("MATMUL_OK=1")
    else:
        print("MATMUL_OK=0")
except Exception as e:
    print("OK=0")
    print("ERR=" + str(e))
PY
) | while IFS= read -r line; do
  case "$line" in
    OK=1) POST_TF_OK=1 ;;
    NGPU=*) POST_TF_GPUS="${line#NGPU=}" ;;
    ERR=*) warn "Post-install TF error: ${line#ERR=}" ;;
  esac
  vecho "$line"
done

# Final NVIDIA guidance if Linux + no nvidia-smi or no GPU
if [ "${OS}" = "Linux" ]; then
  if [ ${have_nvidia_smi} -eq 0 ]; then
    warn "nvidia-smi not available. If you expect NVIDIA GPU, install NVIDIA drivers and CUDA toolkit."
    warn "Admin-required step; see: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html"
  elif [ ${GPU_PRESENT} -eq 0 ]; then
    warn "No GPU reported by nvidia-smi. Ensure drivers are correctly installed and GPU is attached."
  fi
fi

# Summary
printf "\n========================================\n"
printf "Environment Summary\n"
printf "========================================\n"
printf "OS/Arch           : %s / %s\n" "${OS}" "${ARCH}"
printf "Python            : %s (%s)\n" "${PYTHON_BIN}" "${python_ver}"
printf "nvidia-smi        : %s\n" "$([ ${have_nvidia_smi} -eq 1 ] && echo present || echo missing)"
printf "GPU detected      : %s\n" "$([ ${GPU_PRESENT} -eq 1 ] && echo yes || echo no)"
printf "TF import (pre)   : %s\n" "$([ "${TF_IMPORT_OK}" = "1" ] && echo ok || echo fail)"
printf "TF version (pre)  : %s\n" "${TF_VERSION:-n/a}"
printf "TF GPUs (pre)     : %s\n" "${TF_GPUS:-0}"
printf "TF import (post)  : %s\n" "$([ ${POST_TF_OK} -eq 1 ] && echo ok || echo fail)"
printf "TF GPUs (post)    : %s\n" "${POST_TF_GPUS:-0}"

# Exit status logic
EXIT_CODE=0
if [ ${POST_TF_OK} -ne 1 ]; then
  EXIT_CODE=2
  warn "TensorFlow import still failing after fixes."
fi

if [ "${OS}" = "Linux" ] && [ ${have_nvidia_smi} -eq 1 ] && [ ${GPU_PRESENT} -eq 0 ]; then
  EXIT_CODE=3
  warn "nvidia-smi present but no GPU reported; driver/hardware issue likely (cannot auto-fix)."
fi

if [ ${EXIT_CODE} -eq 0 ]; then
  ok "Environment looks good."
  printf "Next steps:\n"
  printf " - If you just installed/changed TensorFlow, restart your Jupyter kernel.\n"
  printf " - Re-run the GPU detection cell in the notebook.\n"
else
  warn "Completed with warnings/errors (exit ${EXIT_CODE}). See messages above."
fi

exit ${EXIT_CODE}
