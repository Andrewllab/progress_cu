#!/usr/bin/env bash
set -euo pipefail
SCIZOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCIZOR_ROOT"
export UV_CACHE_DIR=/tmp/scizor-uv-cache
export UV_PYTHON_INSTALL_DIR="$SCIZOR_ROOT/.python"
if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.11 .venv
fi
uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-robomimic.txt
uv pip install --python .venv/bin/python -e . -e ./robomimic \
    --no-build-isolation --config-setting editable_mode=compat
uv pip check --python .venv/bin/python
echo 'Activate with: source scripts/activate_curation.sh'
