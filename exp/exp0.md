# Experiment 0: ALOHA unbox progress-model training

## Objective

Train SCIZOR progress predictors from the real-robot ALOHA unbox data, using the top camera at the diffusion policy's 50 Hz control rate. Compare predictors trained on clean successful demonstrations (`close`) against predictors trained on `close` plus attempts with setbacks/stalls that ultimately recover (`fail`). Compare the released temporal binning against bins widened for longer real-robot behavior.

## Data preparation and code changes

- Added `scripts/prepare_aloha_progress_hdf5.py` to convert the source camera data into a training-ready HDF5. It samples the `CAM_TOP_orig` stream on a 50 Hz time grid using the latest source frame at or before each target time, resizes frames once to 84x84, and embeds the images directly in the HDF5. This avoids repeated image decoding and opening/indexing source image files during training.
- The prepared file is `runs/aloha_progress/data/aloha_unbox_50hz_84.hdf5`. It contains per-episode `close` and `close_fail` masks. `close_fail` includes both source groups. It contains 120 close episodes and 14 fail episodes (134 total in `close_fail`).
- The prepared HDF5's action field is a placeholder: the progress predictor uses the observation sequence and episode lengths, not actions. **Do not use this HDF5 as the diffusion BC policy dataset.** Prepare/use the original robot actions and observations for BC separately.
- Updated `curation/suboptimal_classifier/dataset/hdf5_dataset.py` so training can read a single HDF5 file, as well as the existing directory-style input, and consume the embedded image arrays.
- Added `curation/suboptimal_classifier/config/aloha_50hz.py` and `aloha_50hz_wide_bins.py`: 50 Hz sampling, 4 transformer blocks, batch size 32, 10,000 training steps, checkpoints every 2,000 steps, and the corresponding bin/rank-threshold definitions below.
- Added `scripts/run_aloha_progress_group.sh` to run each dataset group sequentially through the two bin configurations, with independent checkpoint folders and console logs.

## The four planned runs

| Run name | Training data | Training time bins (seconds) | Intended score-horizon comparison |
|---|---|---|---|
| `goal2_50hz_4layer_close` | `close` | `[0,.5], [.5,1], [1,2], [2,5], [5,∞]` | Score at 2 s and 4 s |
| `goal4_50hz_4layer_close` | `close` | `[0,1], [1,2], [2,4], [4,8], [8,∞]` | Score at 2 s and 4 s |
| `goal2_50hz_4layer_close_fail` | `close_fail` | `[0,.5], [.5,1], [1,2], [2,5], [5,∞]` | Score at 2 s and 4 s |
| `goal4_50hz_4layer_close_fail` | `close_fail` | `[0,1], [1,2], [2,4], [4,8], [8,∞]` | Score at 2 s and 4 s |

The `goal2`/`goal4` run-name tags identify the intended evaluation horizon; **goal time is an inference/scoring setting, not a training parameter**. The training distinction between the two variants is their time-bin configuration. For a clean horizon comparison, evaluate each checkpoint at both horizons rather than assuming its name changes training behavior.

## Launch and output locations

Two tmux sessions are running concurrently. Each session runs its two models sequentially:

- `aloha_progress_close`: close-only models (released bins, then wide bins)
- `aloha_progress_close_fail`: close+fail models (released bins, then wide bins)

Attach with `tmux attach -t aloha_progress_close` or `tmux attach -t aloha_progress_close_fail`; list sessions with `tmux ls`.

The launcher is `scripts/run_aloha_progress_group.sh`. For example, to launch a group manually after its current session has ended, run `bash scripts/run_aloha_progress_group.sh close` or `bash scripts/run_aloha_progress_group.sh close_fail` from the repository root.

- Checkpoints and per-run `config.json`/`metrics.jsonl`: `runs/progress/aloha_unbox/<run_name>/`
- Console logs: `runs/aloha_progress/logs/<run_name>.console.log`
- Training uses local logs; W&B is not required.

At the latest status check, both first (`goal2`) models were training at approximately 3,800/10,000 steps, and GPU memory use was approximately 3.2 GiB total. Once each first model finishes, its session should proceed to that dataset's wide-bin model. The second pair's logs/checkpoint folders appear when those runs start.

## Runtime estimate and interpretation cautions

The concurrent benchmark was about 2.2 steps/s per job, implying roughly 75–90 minutes per 10,000-step model under similar load. Each session has two sequential runs; the two sessions run concurrently, so end-to-end time is approximately 2.5–3 hours, subject to machine load and data-loader/GPU variability. The 16 GB RTX 5060 Ti has ample headroom based on the observed memory use, so these two jobs fit concurrently.

Treat `close` as clean positive examples and `fail` as informative imperfect/recovered behavior, not necessarily as uniformly bad demonstrations. Mixing both changes the predictor's learned ranking signal; inspect per-episode score traces and resulting masks before trusting automated curation. Also verify the desired failure-vs-recovery labeling and inspect representative selected/rejected clips before using filtered data for policy training.

## Useful status checks

```bash
tmux ls
tail -f runs/aloha_progress/logs/goal2_50hz_4layer_close.console.log
tail -f runs/aloha_progress/logs/goal2_50hz_4layer_close_fail.console.log
find runs/progress/aloha_unbox -maxdepth 2 -type f | sort
```
