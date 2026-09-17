#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != close && "$1" != close_fail ) ]]; then
  echo "Usage: $0 {close|close_fail}" >&2
  exit 2
fi

SCIZOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$SCIZOR_ROOT/runs/aloha_progress/data/aloha_unbox_50hz_84.hdf5"
GROUP="$1"
FILTER="$GROUP"
LOGDIR="$SCIZOR_ROOT/runs/aloha_progress/logs"
mkdir -p "$LOGDIR"

if [[ ! -f "$DATASET" ]]; then
  echo "Missing prepared HDF5: $DATASET" >&2
  exit 1
fi

source "$SCIZOR_ROOT/scripts/activate_curation.sh"
cd "$SCIZOR_ROOT"

run_one() {
  local horizon_tag="$1"
  local config_path="$2"
  local run_name="goal${horizon_tag}_50hz_4layer_${GROUP}"
  local log_path="$LOGDIR/${run_name}.console.log"
  echo "Starting $run_name at $(date --iso-8601=seconds)"
  python -m curation.suboptimal_classifier.train_hdf5 \
    --config="$config_path" \
    --name="$run_name" \
    --config.hdf5_dataset_kwargs.data_dir="$DATASET" \
    --config.hdf5_dataset_kwargs.filter_key="$FILTER" \
    --config.hdf5_dataset_kwargs.obs_keys.agentview_image=84 \
    2>&1 | tee "$log_path"
}

# goal_time is used at scoring time, not during predictor training. These two
# runs use different training bins; each checkpoint can later be scored at 2 s
# and 4 s for a true horizon comparison.
run_one 2 curation/suboptimal_classifier/config/aloha_50hz.py
run_one 4 curation/suboptimal_classifier/config/aloha_50hz_wide_bins.py
