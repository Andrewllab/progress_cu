# Source this file from bash: source scripts/activate_curation.sh
SCIZOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCIZOR_ROOT/.venv/bin/activate"
export PYTHONPATH="$SCIZOR_ROOT/robomimic:$SCIZOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCIZOR_ROOT/.cache/huggingface"
export TORCH_HOME="$SCIZOR_ROOT/.cache/torch"
export NUMBA_CACHE_DIR="$SCIZOR_ROOT/.cache/numba"
export MPLCONFIGDIR="$SCIZOR_ROOT/.cache/matplotlib"
export UV_CACHE_DIR=/tmp/scizor-uv-cache
export TOKENIZERS_PARALLELISM=false
