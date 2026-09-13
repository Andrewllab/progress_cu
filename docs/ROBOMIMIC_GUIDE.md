# SCIZOR for robomimic and your real robot

This guide describes the checked-out implementation and the local execution path
added during setup. It is not a claim that the released implementation exactly
reproduces every equation in the paper. No experiment on your real data has run.

## Environment and scope

From the repository root:

```bash
source scripts/activate_curation.sh
```

The environment is `.venv`, with Python 3.11, PyTorch 2.7.1+cu128,
torchvision 0.22.1, transformers 4.46.3, NumPy 1.26.4, FAISS CPU 1.8.0,
the editable **bundled** robomimic fork, robosuite 1.4.1, MuJoCo 2.3.7,
and the remaining packages in `requirements-robomimic.txt`.
The host has an RTX 5080 (16 GB), driver 580.119.02. GPU access requires
running outside the restricted execution sandbox. The CUDA 12.8 build was
selected for this GPU; the README's CUDA 11.8/PyTorch 1.13 recipe is unsuitable
for this hardware. See [PyTorch's version-specific installation commands](https://pytorch.org/get-started/previous-versions/).

`bash scripts/setup_curation.sh` reconstructs the environment using `uv`.
The activation helper sets project-local model/Numba caches and puts the bundled
robomimic on the import path. Do not replace this fork with stock pip robomimic:
its dataset loader contains the curation integration.

DINOv2 and the original Cosmos CV8x16x16 encoder are downloaded to `.cache/`.
`python scripts/download_curation_models.py` downloads them again if needed.
The Cosmos HDF5 encoder now loads the TorchScript encoder directly and uses its
`encoder` and `quant_conv` submodules, as the original SCIZOR code did after
loading the external wrapper. No Cosmos text/video generation models are needed.
The [original NVIDIA tokenizer implementation](https://github.com/NVIDIA/Cosmos-Tokenizer)
and [checkpoint](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-CV8x16x16)
are the relevant upstream components.

Octo, JAX/Flax, TensorFlow, TFDS, DALI, LanguageBind and GroundingDINO are not
installed. The original OXE/distributed launchers still require their own stack.
FAISS clustering uses CPU; PyTorch model inference and local pairwise similarity
can use CUDA. This is a practical single-workstation setup, not a benchmark of
the original distributed infrastructure.

## What the project does

The paper proposes two complementary filters: temporal-progress prediction for
locally inefficient behavior, and visual/action similarity for redundant chunks.
It uses temporal bins, distributes segment scores to transitions, blends local
and trajectory scores, and excludes samples during policy training. Its appendix
describes frozen DINOv2 and feature differences, and reports thresholds 0.58 and
0.99. These are reference settings, not guarantees for a new camera, robot or task.
Source: [SCIZOR, Sections 3.2–3.4 and Appendix A](https://arxiv.org/pdf/2505.22626).

The concrete data flow in this checkout is:

```text
HDF5 image trajectories
  ├─ random pairs + elapsed-time labels → DINOv2 progress classifier
  │    → checkpoint/config → overlapping-window inference → subop_score[T]
  └─ fixed non-overlapping chunks → Cosmos features + sampled actions
       → features.pkl → clustering + cosine comparisons → dedup.pkl
       → max_sim[K], max_sim_idx[K,2]

original HDF5 + score arrays → bundled robomimic SequenceDataset
  → combine removal masks → sample retained anchors → BC policy training
  → simulation success-rate evaluation
```

This does **not** relabel actions, repair trajectories, or train a reward model
from success labels. Filtering changes the sampling of the existing demonstration
data. A slow movement can be intentional (contact, insertion, settling), so a
high progress discrepancy needs qualitative inspection before aggressive removal.

## Repository map

| Component | Location | Role |
|---|---|---|
| Progress settings | `curation/suboptimal_classifier/config/config.py` | Shared configuration; defaults include older action/GroundingDINO experiments |
| Released robomimic settings | `curation/scripts/subop_train_robomimic.sh` | Overrides shared defaults for DINOv2 image-pair classification |
| Direct local training (added) | `curation/suboptimal_classifier/train_hdf5.py`, `config/robomimic.py` | Single-device HDF5 training; compatible config/checkpoints |
| Released distributed trainer | `accelerate_train_one_loader.py` | Accelerate workers consume batches from ZMQ producer processes |
| Legacy trainer | `train.py` | Older action-augmentation path; not the entry point for your task |
| Pair sampler | `dataset/hdf5_dataset.py` | Reads demonstrations, samples image pairs and temporal labels |
| OXE input stack | `dataset/dataset.py`, `dataset/rlds/`, `dlimp/` | RLDS transforms, action augmentation, DALI, ZMQ and dataset mixtures |
| Progress architecture | `discriminator/discriminator.py` | Encoder, temporal embeddings, attention blocks, rank head, loss/metrics |
| Offline scoring | `curation/video_encoding/suboptimal_hdf5.py` | Loads checkpoints, predicts bins, creates transition scores |
| Video/action encoding | `video_encode_cosmos_hdf5.py` | Chunk IDs, Cosmos representations and action arrays |
| Dedup machinery | `curation/semdedup/` | Feature loading, clustering, ordering, similarity, aggregation |
| Local dedup runner (added) | `curation/semdedup/local.py` | Reuses feature normalization and similarity code without SLURM |
| Score attachment | `write_semdup_score.py` | Attaches chunk scores/intervals to HDF5 |
| Policy integration | `robomimic/robomimic/utils/dataset.py` | Thresholding and retained-sample indexing |
| Policy execution | `robomimic/robomimic/scripts/train.py`, `utils/train_utils.py` | BC training, validation and environment rollouts |
| Policy templates | `robomimic/robomimic/exps/curation_exps/robomimic/{can,square}/bc_curation.json` | Image BC-GMM configurations with curation controls |
| OXE policy experiments | `octo/` | Customized Octo data loading and training; outside this setup |
| Analysis tools | `curation/visualization/`, `curation/utils/` | Plots, inspection, OXE utilities; some still require optional dependencies |

## Progress training: what a training example actually is

`Hdf5SubDataset` measures each demo's length. The outer index chooses a demo
proportionally to its number of frames; it does **not** deterministically select
that frame as the start. `get_rand_dist()` randomly selects one of five bins,
samples a duration inside that bin, converts seconds to frames using `freq`,
chooses a start frame, and clips the end to the trajectory boundary.

Its target is the **actual resulting time gap** `(end-start)/freq`, not an
annotated quality score. The bins are `[0,.5], [.5,1], [1,2], [2,5], [5,10000]`.
The final interval is a finite stand-in for an open-ended bin. Short trajectories
and clipping can change the class distribution even though bins are sampled
uniformly. The released sampler is preserved; inspect label histograms when
changing control frequency or episode length.

Images are uint8 RGB in `[T,H,W,3]`, scaled by 255, normalized by ImageNet mean/std,
permuted to channels-first and resized. The local default reads only
`agentview_image`; the released HDF5 defaults load both cameras but the batch
interface selects just `discriminator_dataset_kwargs.image_key`. Progress training
is **single-view**, even though the BC policy can use both cameras.

At batch size B the model receives `[B,2,3,84,84]`. Frozen
`facebook/dinov2-base` produces token features of width 768 for each image.
Temporal positional embeddings distinguish the two observations. Six attention
blocks update a learned query/CLS representation, and the rank head returns five
probabilities. There is no action or language input in the robomimic configuration.

The new direct trainer uses AdamW and OneCycleLR, writes local JSONL metrics,
and saves `config.json` plus `model_<step>.pth`. It deliberately retains the
released model and loss. It does not implement distributed training, W&B,
automatic evaluation, mixed precision, optimizer-state resume or accumulation.
`load_path` loads weights only. Shared `eval_interval` fields do not activate
evaluation in this entry point. Use held-out demonstrations and rollouts to
evaluate experimental changes.

| Setting | Released robomimic value | Local default / effect |
|---|---|---|
| `hdf5_dataset_kwargs.freq` | 20 Hz | 20; must equal actual sampled observation rate |
| `hdf5_dataset_kwargs.batch_size` | 128 | 32 for a 16 GB workstation; tune after measuring memory |
| `hdf5_dataset_kwargs.num_workers` | 4 | 0, avoiding inherited open HDF5 handles |
| `hdf5_dataset_kwargs.filter_key` | Not supported originally | Optional `mask/train` selection added |
| `window_size`, `future_image` | 1, true | Current/future observation pair |
| `encoder_type`, `frozen_encoder` | dinov2, true | Pretrained visual features; head learns temporal relation |
| `d_model`, `num_blocks`, `n_heads` | 768, 6, 8 | Attention capacity |
| `head_type`, `loss_fn_type` | rank, cross_entropy | Five-bin classification; caveat below |
| `no_action_input`, `no_text_input` | true, true | Do not load action/language encoders |
| `optimizer.lr`, `weight_decay` | 1e-4, 1e-3 | Optimizer settings |
| `num_steps`, `save_interval` | 10000, 2000 | Training duration and checkpoint cadence |

Changing batch size changes total examples seen at a fixed number of steps.
For matched experiments, record both steps and effective examples processed.
If changing bins, edit **both** `hdf5_dataset_kwargs.time_bins` and
`discriminator.rank_thres`; matching defaults do not imply they stay synchronized.

## Discriminator architecture and tensor shapes

The following is the exact default shape flow for the local robomimic
configuration (`encoder_type=dinov2`, `window_size=1`, `future_image=true`,
`no_action_input=true`, `no_text_input=true`, `head_type=rank`). `B` is batch
size, `T=2` is current plus future image. With SCIZOR's 84×84 resize and
DINOv2's 14-pixel patch size, each image has 37 tokens (one CLS token plus a
6×6 patch grid) of width 768. The standard DINOv2 224×224 preprocessing would
instead produce 257 tokens, but this code bypasses that processor.

```text
HDF5 pair sampler
  current RGB, future RGB
       [B, 2, 3, 84, 84]
              │
              ├─ split along T; two [B, 3, 84, 84] images
              │
              ├─ frozen DINOv2-base, independently per image
              │       two [B, 37, 768] feature sequences
              │
              ├─ stack temporal image features
              │       [B, 2, 37, 768]
              │
              ├─ flatten image-token axes
              │       [B, 74, 768]
              │
              ├─ learned action-query bias (because no_action_input=True)
              │       [B, 1, 768]
              │     + learned CLS token
              │       [B, 2, 768]
              │
              ├─ concatenate image tokens and query tokens
              │       [B, 76, 768]
              │
              ├─ six FeatureFusionBlock self-attention blocks
              │       [B, 516, 768] → [B, 516, 768]
              │
              ├─ select token 0 (the learned CLS token)
              │       [B, 768]
              │
              ├─ linear rank head: 768 → 5
              │       [B, 5]
              │
              └─ softmax probabilities for the five temporal bins
                      [B, 5] → reshaped as model output [B, 1, 5]
```

The image feature extractor is frozen, while the learned query, positional
embeddings, fusion blocks and rank head are trainable. In this configuration the
query is not an encoded action; it is a learned token that gives the transformer
a prediction slot. The `action_query_length=1` setting means one such query
before the CLS token. `head_token="cls"` makes the head read token zero.

The six self-attention blocks are implemented by `FeatureFusionBlock` and
`AttentionBlock`. Each block performs multi-head self-attention over the
concatenated image/query sequence, followed by residual LayerNorm and a
two-layer GELU feed-forward network. The robomimic launcher overrides the base
configuration from `cross-attn` to `self-attn`; therefore the default robomimic
model does not run a separate image-to-query and text-to-query cross-attention
stage. The base configuration's cross-attention option is still available for
other experiments.

The model's public call is `Discriminator.forward(image, text, action, score=...)`.
For training, `image` is `[B,2,3,H,W]`, `text=None`, `action=None`, and `score`
is the sampled elapsed time `[B]` or `[B,1]`. The forward method reshapes this
to `[B,1]`, produces `output=[B,1,5]`, bucketizes the continuous time using
thresholds `[0.5, 1.0, 2.0, 5.0]`, and computes a scalar cross-entropy loss.
The returned dictionary contains `output`, `loss`, and metric values. At exact
boundaries, the current `right=False` bucketization places the value in the
lower-index convention used by this code path.

At inference with one lookahead, `output` is `[B,1,5]` and the scoring script
turns it into a five-class probability array before calculating the progress
discrepancy. If several lookaheads are requested, the input is logically
`[B,N,2,3,H,W]`, internally flattened to `B×N` image pairs, and the output is
`[B,N,5]`; scores are then averaged over `N` horizons.

There are two important non-default paths. With action input enabled, the action
sequence is projected into one or more learned query tokens, with action and
frame positional embeddings, and those queries are fused with the image tokens.
With text enabled, a frozen DistilBERT (or GroundingDINO's text branch) supplies
additional tokens. Those paths change the sequence length but still end in the
same CLS-or-mean pooling and regression/rank head. They are not used by the
released robomimic progress experiment.

The rank head's five outputs correspond to the temporal bins, not five labels
such as success, failure, pause, or operator skill. The code currently applies
softmax inside `RankHead` and then passes that result to PyTorch
`CrossEntropyLoss`, which normally expects logits. This is a reproducibility
caveat in the released implementation; it is one reason to treat the output as
an implementation-specific calibrated score when tuning thresholds.

## Inference, scores and filtering

`load_model()` reconstructs the architecture from the checkpoint's neighboring
`config.json`. A directory selects its numerically latest `.pth` checkpoint;
passing an explicit checkpoint is preferable for reproducible comparisons.

For every start frame, `PreprocessDataset` pairs the current observation with one
or more future observations. `--goal-time 2` means a 40-frame lookahead at 20 Hz,
clipped at the episode end. It can accept comma-separated horizons, e.g. `1,2,4`.
The model output is `[B,number_of_horizons,5]`.

The implementation in `rank_prob_to_score()` does the following:

1. Computes expected **bin index**, `r_pred = sum(k * p_k)`.
2. Converts the actual time gap into an interpolated bin coordinate `r_actual`:
   within bin k, `(dt-lower)/(upper-lower) + k - 0.5`, clipped to `[0,4]`.
3. Computes `q = clip((r_actual-r_pred)/max(frame_gap,1), 0, 1)`.
4. Convolves q over a lookahead-sized window and divides by the contributing
   window count to spread the discrepancy across transitions.
5. Accumulates backward: `score[j-1] += gamma**(1/freq) * score[j]`.
   The released loop stops before updating index 0. Gamma defaults to 0.5;
   `--discount-gamma` now exposes it without changing the default.
6. Averages across requested horizons and writes `data/demo_*/subop_score`.

For a full two-second window, `r_actual=2.5`. If `r_pred=1`, its initial q is
`1.5/40 = 0.0375`, before overlap aggregation and backward accumulation.
This is not an estimate of "probability this action fails" and is not simply
elapsed seconds minus predicted seconds. More accumulation can increase its scale.

During policy dataset construction, `SequenceDataset` optionally smooths these
scores and applies `mix_level * mean(trajectory_score) + (1-mix_level) * local_score`.
With `mix_level=.5`, local and trajectory evidence contribute equally.
The scorer CLI's `--mix-level` affects its visualization; it does **not** change
the raw scores written to disk. The policy's mixture is the one affecting training.

| Policy control | Exact meaning in this fork |
|---|---|
| `subop_curate.enabled` | Activate suboptimality mask |
| `subop_percentile=.707` | Remove scores strictly above the 70.7th percentile; approximately 29.3% removed, subject to ties |
| `percentile_as_threshold=true` | Interpret `subop_percentile` as a raw score cutoff instead, despite the name |
| `smooth_window` | Moving-average length in frames |
| `mix_level=0/.5/1` | Local-only / mixture / trajectory-mean score |
| `traj_level=true` | Give every transition its trajectory mean |
| `dedup_type=semantic`, `keep_ratio=.997` | Keep the lowest-similarity 99.7% of **chunks** |
| `dedup_type=threshold`, `keep_ratio=.99` | Here `keep_ratio` is a cosine cutoff: keep chunks with `max_sim < .99` |

The suboptimal and dedup masks are combined with OR. Report the union's actual
removal rate; adding the two percentages double-counts overlap. Chunk retention
is not exactly transition retention because final chunks can have different lengths.

Filtering selects valid sequence **anchors**; observation history and future
actions are fetched from original contiguous trajectories. Do not physically
concatenate surviving timesteps: that would invent discontinuous transitions.
The supplied feed-forward `seq_length=1` setting is the best starting point;
audit the fork's padding/goal-index behavior before using unpadded long sequences.

## Deduplication details

`Hdf5Dataset` in the Cosmos encoder constructs non-overlapping two-second chunks
(40 steps at 20 Hz), keeping the shorter last chunk. Eight evenly spaced images
and action vectors are sampled per chunk. Images become `[B,3,8,128,128]`.
The checked encoder emits a `[B,16,2,8,8]` latent, flattened into a 2048-vector.
Actions are padded to 14 dimensions and sampled to `[8,14]`.

`features.pkl` stores `id`, `image_embeds`, `action`, and `normalized_action`.
The latter uses per-file mean/std. The **released default uses raw `action`**.
`EmbeddingLoader` L2-normalizes each modality and scales it by the square root
of its weight, then concatenates them. With image/action weights .8/.2, the dot
product is approximately `.8*cos(video) + .2*cos(action)`.

Spherical k-means makes groups; within each group, ordering chooses which samples
get priority (`random`, `hard`, `easy`). The upper-triangular similarity calculation
assigns a chunk its maximum similarity to **earlier ordered chunks**, not an
unrestricted nearest neighbor that could eliminate every member of a duplicate
group. The original cosine implementation clamps the effective minimum to zero
through triangular zero entries. The local runner preserves that behavior.
Representatives are not chosen according to the progress score.

The old pipeline is `clustering.py → sort_clusters.py → semdedup.py →
concat_cluster_df.py`; the new `local.py` combines those operations for a
workstation and writes the HDF5 writer's expected indexed DataFrame directly.
It uses FAISS CPU k-means and can use CUDA for pairwise comparison. Pairwise
memory grows quadratically with the largest cluster; full features also live in
RAM. Choose a cluster count suited to the number of chunks. The original 5000
centroids / eight-GPU generator default is inappropriate for a small trial.

The old generated `eps=...` columns actually use **per-cluster percentiles**, not
`1-eps` cosine cutoffs as their comments suggest. These columns are not the raw
`max_sim` threshold used by the HDF5 policy path. The local path writes raw scores.

## Run on your data

Use a working copy: both score writers modify HDF5 in place and overwrite their
score fields on reruns. Keep only intended HDF5 files under each input directory;
the scripts recurse into subdirectories. Deduplicate separately per task/action
space, with unique dataset names. For strict held-out evaluation, prepare a
training-only HDF5 copy before encoding: the encoder does not select `mask/train`.

Training/scoring example (replace paths; commands assume repo root and activation):

```bash
python -m curation.suboptimal_classifier.train_hdf5 \
  --name can_progress_seed42 \
  --config.hdf5_dataset_kwargs.data_dir=/absolute/path/work/can \
  --config.hdf5_dataset_kwargs.filter_key=train \
  --config.hdf5_dataset_kwargs.batch_size=32

python -m curation.video_encoding.suboptimal_hdf5 \
  --data-dir /absolute/path/work/can \
  --model-path runs/progress/can_progress_seed42/model_10000.pth \
  --goal-time 2 --discount-gamma 0.5 --batch-size 32 --save-score

python -m curation.video_encoding.video_encode_cosmos_hdf5 \
  --data-dir /absolute/path/work/can \
  --output-dir runs/embeddings/can --cosmos-path .cache/cosmos \
  --batch-size 4 --chunk-time 2 --freq 20

python -m curation.semdedup.local \
  --embedding-dir runs/embeddings/can --output-dir runs/dedup/can \
  --ncentroids 50 --image-weight 0.8 --which-to-keep random

python -m curation.video_encoding.write_semdup_score \
  --data-dir /absolute/path/work/can \
  --semdedup-path runs/dedup/can/dedup.pkl

python scripts/make_policy_config.py \
  --dataset /absolute/path/work/can/mh/image.hdf5 \
  --output runs/configs/can_both.json --task can --mode both

python -m robomimic.scripts.train --config runs/configs/can_both.json
```

The policy config helper accepts `none`, `subop`, `dedup`, `both`; it reduces
rollouts to 20 sequential environments and reads `mask/train`. It disables the
fork's curated validation loss and uses rollout evaluation. For offline-only
experiments use `--no-rollouts`; evaluation then needs a separate protocol.
Data without masks needs masks created first or a deliberate change to the
generated `hdf5_filter_key`. Generated output directories/configs must be new.

The original Can and Square templates have **dedup disabled**, despite their
curation names. They are BC-GMM, not BC-RNN or a diffusion policy. Their original
policy training budget is 600 epochs × 500 updates, batch size 16.

## Implementation caveats before tuning

These were found by reading the actual execution path:

- `RankHead.forward()` applies softmax, then `compute_loss()` passes those
  probabilities into `CrossEntropyLoss`, which expects logits. This remains
  unchanged to preserve the released objective; use a clearly named new variant
  if correcting it, and keep inference probabilities consistent with checkpoints.
- The architecture computes delta features, but the default branch discards them
  and keeps the original two-image token sets. `cat_start_feature=true` enables
  start-plus-delta features; it is not a pure delta-only architecture. This differs
  from the appendix description. Also, each fusion block receives the original
  image tokens; updated image tokens are not carried to the next block.
- Training uses `torch.bucketize(..., right=False)`: exact boundaries such as
  2.0 seconds go into the lower class. Scoring uses half-open intervals and an
  interpolated rank. Treat boundary conventions consistently in a corrected variant.
- The scoring code uses bin units, overlap **averaging**, and **backward** discount
  propagation. The paper's written temporal sum propagates past terms forward;
  do not assume numerical equivalence. The code also leaves index 0 unpropagated.
- HDF5 progress sampling has no task/language conditioning or balanced task
  mixture, and the original loader ignores train/valid masks. The local loader
  adds optional mask selection, but does not add a held-out evaluator.
- The released policy loader applies curation to its validation dataset as well.
  Comparing that validation loss across thresholds confounds the validation set
  with the intervention. The provided local config helper avoids that metric.
- Cosmos preprocessing here uses `[0,1]` input, whereas NVIDIA documents `[-1,1]`
  for its tokenizer API. This setup preserves SCIZOR preprocessing; changing it
  requires re-encoding and recalibrating similarity cutoffs.
- Original dedup script generation still contains `semadedup` path typos and
  cluster/SLURM assumptions; use the tested local runner. Some visualization
  code still assumes contiguous `demo_0`, `demo_1`, ... and intervention labels.
  The generic scoring path works without intervention labels, but arbitrary demo
  names need extra care when requesting videos. The dedup writer matches dataset
  names by substring; use isolated task directories to avoid ambiguous IDs.

Setup fixes cover packaging, unnecessary optional imports, positional encoding,
CPU fallback, zero-worker loading, noncontiguous names in progress training,
one-frame/short-trajectory score shapes, and the local execution routes.
The scientific objective, default temporal sampler, default feature choice,
normalization and discounting remain those of the release.

## Suggested experiment order

These are recommendations for your investigation, not claims of measured gains:

1. Start with Can MH, then Square MH, using the correct simulator/data version.
   Save unfiltered BC results, rollout seeds, success rates and confidence intervals.
   Verify dataset `env_args` before rendering/replaying older releases; this fork's
   download registry still contains older robomimic links. A smoke test of robosuite
   1.4.1 does not establish compatibility with every legacy dataset.
2. Train progress on training demonstrations only. Inspect predicted classes,
   label frequencies, per-bin accuracy and videos from both ends of the score
   distribution. Keep held-out episodes/operators separate, not random frames.
3. Initially fix the classifier and sweep suboptimal retention `.95,.90,.80,.707`,
   mixture `0,.5,1`, and lookahead `1,2,4` seconds. Change one axis first. Reuse
   stored scores for mixture/threshold sweeps; changing lookahead/gamma requires
   rescoring, not retraining. Changing bins or architecture requires retraining.
4. Start dedup conservatively, comparing no removal with about `.3%,1%,5%`
   chunk removal. Sweep image/action weights `.5/.5,.8/.2,.95/.05`, inspect
   candidate pairs, and repeat cluster-order seeds. Re-cluster when changing
   modality weights or action representation. Raw and standardized actions are
   distinct variants (`--action-key normalized_action` selects the latter).
5. Run the four component ablations and matched-size random-removal controls.
   Keep policy update budgets matched and use at least three policy seeds for
   promising settings. Report retained transitions, retained chunks, dropped
   episodes, removal by task/stage/operator, and overlap between filters.
6. Only then compare scientific corrections (logits-based cross-entropy,
   delta features, consistent boundaries/discounting) against the retained release
   behavior. They may change calibration; old cutoffs should not be carried over.

## Real-robot data contract

For progress training, provide `data/demo_*/obs/<camera>` as RGB uint8
`[T,H,W,3]`, `data/demo_*/actions` as `[T,A]`, and `num_samples=T` per demo.
The progress model uses the actions array to establish length, not as model input.
The policy loader additionally expects the configured `next_obs`, proprioception,
`rewards` and `dones` fields; the progress/dedup minimum is not the full BC schema.
For dedup, actions must have consistent units, semantics and dimension within a
run (`action_max_dim` defaults to 14). The camera key must match both training
and inference. Frequency is currently a single configuration value, not inferred
per episode: variable-rate recordings should be resampled/aligned first.

Preserve timestamps, camera identity, language/task labels, operator/session IDs,
success/intervention annotations and action convention in metadata when available.
Use those fields for splitting/auditing even though the current progress model
doesn't consume them. For real-robot BC, also map proprioception and observation
keys, configure action normalization, and define evaluation for your hardware;
the Can/Square rollout configuration is not a real-robot evaluator.

When the dataset arrives, first audit synchronization, control frequency, image
orientation and action conventions, then build the adapter and splits. Contact
waiting, recovery attempts and pauses should be inspected separately: removing
them indiscriminately can erase useful recovery or precision behavior.

## Verification completed during setup

- Dependency consistency check passed; the exact installed versions are captured
  in `requirements-robomimic.lock.txt` (editable local packages installed separately).
- CUDA matrix computation passed on the RTX 5080 with native `sm_120` support.
- Downloaded DINOv2 trained for two steps on synthetic HDF5 data using the six-block
  configuration, then saved and reloaded a checkpoint for complete trajectory scoring.
- Downloaded Cosmos encoder produced finite features and encoded the synthetic
  HDF5 dataset. Local clustering/similarity and the HDF5 score writer completed.
- Bundled image BC-GMM trained for six updates with **both** curation masks enabled;
  it retained 60 of 96 synthetic training transitions. No policy performance claim
  follows from these artificial observations/actions.
- Robosuite Lift reset, stepped and rendered an 84×84 camera image with MuJoCo/EGL.
- `python -m pytest tests/test_curation_local.py -q`: three tests passed, covering
  masked/noncontiguous HDF5 loading, short-window score shapes, and duplicate score
  attachment. Changed entry points passed Python compilation and `git diff --check`.

Synthetic data/checkpoints are under `.cache/smoke/`; BC debug logs are under
`/tmp/tmp_trained_models/scizor_smoke_verified/`. Environment/model setup and the
local path are verified; benchmark reproduction and real-robot adaptation await
the actual datasets. This setup made no commits and changed no user datasets.
