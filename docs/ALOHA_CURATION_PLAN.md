# ALOHA progress-model training and data-curation plan

Recorded: 2026-09-16. Dataset and code inspection performed during the preceding analysis.

This document preserves the findings and recommendations from that analysis. It is a design proposal, not an implementation or an experimental result. No training, dataset conversion, or curation was performed during the analysis.

## Summary

Use SCIZOR as a detector of local inefficiency, initially train it on verified successful demonstrations, and adapt its filtering to entire diffusion-policy action chunks. Changing only the recording frequency in the existing configuration is insufficient.

Recommended starting point:

- Resample progress observations onto a timestamp-based **10 Hz** grid. Keep a separate policy dataset at the confirmed **50 Hz** deployment rate.
- Compare the released temporal bins against **0–1, 1–2, 2–4, 4–8, and ≥8 seconds**.
- Evaluate **2-, 4-, and 8-second** lookaheads separately before combining calibrated scores.
- Train first on verified successful training episodes, with episode-balanced and feasible-bin-balanced sampling.
- Use frozen DINOv2, initially the top camera at **140×140**, two fusion blocks, and batch size 16.
- Correct and version the logits-based loss and temporal score construction before relying on thresholds.
- Begin with local scores (`mix_level=0`), conservative removal, and explicit preservation of useful contact, recovery, and terminal behavior.
- Integrate masks into the diffusion policy's **supervised action chunks**, preserving original temporal continuity.
- Evaluate curation against successful-only, mixed-data, and matched-size random-removal policy baselines.

These settings are hypotheses to validate, not measured optima.

## 1. Scope, evidence, and clarified task semantics

Inspected sources:

- [Local Robomimic guide](ROBOMIMIC_GUIDE.md).
- [SCIZOR paper](https://arxiv.org/html/2505.22626v2).
- The local progress sampler, classifier, direct HDF5 trainer, scorer, and Robomimic sequence-filtering implementation.
- Timestamps across all 134 episodes in:
  - `/home/weiran/projects_wr/datasets/aloha_unbox/close`
  - `/home/weiran/projects_wr/datasets/aloha_unbox/fail`
- Pose-array lengths and finite values across those episodes, plus representative camera images.

User clarification (2026-09-16): `close` contains successful trajectories without obvious progress regression or minor failures. `fail` contains trajectories with local failures that are recovered from, potentially including regression and stalls. It is not an episode-level unsuccessful-outcome label. The diffusion policy will use **50 Hz** control.

The diffusion-policy implementation and precise action representation remain unspecified. In particular, confirm whether commands are 14 joint-position targets and how leader joints and grippers map to follower commands. Control frequency is confirmed as 50 Hz.

### Updated recommendations following clarification

These points supersede any earlier wording below that describes `fail` episodes as terminally unsuccessful:

- Use clean `close` training episodes as the initial progress reference. Do not assign negative labels to every transition in `fail` or remove that group wholesale.
- Preserve useful recovery for policy training. A clean-only progress model may be unfamiliar with recovery states; uncertainty or temporary regression should trigger review rather than automatic deletion.
- Compare policy baselines using `close` only, `close` plus all `fail`, and `close` plus locally curated `fail`. Evaluate recovery preservation explicitly.
- Keep trajectory mixing at `mix_level=0` initially to avoid penalizing useful recovery because another interval contains a stall.
- Keep progress observations at the proposed 10 Hz, but construct a separate **50 Hz policy grid from timestamps**. The measured roughly 45 Hz recordings must not simply be relabeled as 50 Hz.
- For policy images, use the latest available frame and record its age; repeated frames are expected. Align state and commands according to their physical meaning. Whether to hold or interpolate joint targets depends on controller and logger semantics. Resampling cannot recover events lost during the observed gaps of up to 87 ms.
- Map rejected time intervals onto 50 Hz action targets by interval overlap. A one-to-five expansion is valid only after both grids are aligned.
- At 50 Hz, 16 actions span 0.32 seconds, 32 span 0.64 seconds, and 50 span one second. Choose prediction and execution horizons separately from progress lookaheads.
- Audit chunk rejection at setback-to-recovery boundaries: rejecting an entire chunk can remove useful recovery targets alongside a short bad interval.
- Treat collection date as a potential confound for clean-versus-setback group identity, not as evidence of terminal success versus failure.

The visual inspection was a spot check, not a full behavioral annotation. Failure durations below are episode durations, not measured durations of individual failure behaviors.

## 2. Dataset findings

| Measurement | `close` | `fail` |
|---|---:|---:|
| Episodes | 120 | 14 |
| Recorded timesteps | 118,104 | 30,016 |
| Episode duration, minimum / median / maximum | 11.1 / 21.3 / 37.9 s | 26.8 / 44.6 / 68.8 s |
| Timesteps per episode, median | 960 | 1,997 |
| Total recording time | 43.5 min | 11.2 min |

Across both groups:

- Episode-average rates range from **42.0 to 46.7 Hz**, with median **45.4 Hz**.
- Timestamps are strictly increasing but irregular; the largest observed gap is approximately **87 ms**.
- All three camera streams have matching timestamp sequences and frame counts. JPEG filenames agree with leader timestamps to rounding precision.
- Leader/follower timestamp offsets reach **8.7 ms**. Logger timestamp agreement does not establish camera exposure synchronization or control latency.
- Joint positions have shape `[T,14]`; end-effector poses have shape `[T,4,4]`.
- All 804 inspected pose arrays had episode-consistent lengths and finite values.
- Representative top and left images are 384×384; representative right images are 224×224. Image dimensions were not exhaustively checked for every frame.
- Data is stored as timestamped JPEG files and separate leader/follower `.pt` tensors, not the HDF5 schema required by the existing trainer.

### Consequences

Do not treat these recordings as fixed 50 Hz. That would underestimate elapsed time and misalign temporal settings. Build any regular grid from timestamps and retain source-index mappings.

`fail` accounts for only **10.4% of episodes**, but **20.3% of timesteps**. The current frame-weighted sampler gives these longer episodes disproportionately large influence.

There is a collection-date confound: failure recordings are from June 24–25, while `close` recordings come from other dates. Folder-separation performance could reflect session appearance rather than behavior quality. Use session-aware evaluation and avoid interpreting folder classification as proof of progress understanding.

## 3. Robomimic reference and temporal scaling

SCIZOR's detailed settings specify five temporal-distance bins and a two-second inference interval. Its appendix also contains a conflicting statement about five equal bins; the explicit intervals and this checkout agree on the unequal bins below. See [SCIZOR Appendix A](https://arxiv.org/html/2505.22626v2#A1).

| Class | Released interval | Approximate frames at 20 Hz | Proposed ALOHA interval |
|---|---|---:|---|
| 0 | 0–0.5 s | 0–10 | 0–1 s |
| 1 | 0.5–1 s | 10–20 | 1–2 s |
| 2 | 1–2 s | 20–40 | 2–4 s |
| 3 | 2–5 s | 40–100 | 4–8 s |
| 4 | ≥5 s | ≥100 | ≥8 s |

These classes represent **elapsed time between two observations**, not five task stages or five quality categories. Boundary conventions must be made consistent in a corrected implementation.

Published Robomimic statistics report average trajectory lengths of **209 steps for Can-MH** and **269 steps for Square-MH**. At this checkout's 20 Hz convention, these correspond to approximately **10.5 and 13.5 seconds**. These are reference dataset statistics, not measurements of the exact files used by SCIZOR. Source: [ICLR 2024 study, Table 13](https://proceedings.iclr.cc/paper_files/paper/2024/file/852f50969a9e523ec41d26f2f68bd456-Paper-Conference.pdf).

A two-second window spans approximately:

- 15–19% of those Robomimic reference durations.
- 9% of the median `close` episode.
- 4.5% of the median `fail` episode.

Successful ALOHA episodes are roughly twice as long in physical time, while their frame counts are several times larger. **Recording frequency and task duration need separate treatment.**

### Separate progress and policy rates

Start with a **10 Hz progress observation grid**, using timestamps. This gives 20, 40, and 80 observations over two-, four-, and eight-second windows. Compare against 20 Hz if brief contact events are missed.

Use the confirmed **50 Hz policy rate**. Do not reduce policy frequency merely because progress estimation uses 10 Hz. Transfer scores between datasets by time and episode identity, not by assuming equal array indices.

## 4. Algorithm recommendations

### 4.1 Learn a reference for useful behavior

Elapsed time is only a proxy for task progress. If struggling for ten seconds is common in training, the classifier can learn that struggle as normal temporal evolution.

For the first model:

- Use verified successful **training** episodes from `close`.
- Sample episodes uniformly, then feasible temporal bins, then valid pairs.
- Exclude recording setup, resets, and long post-completion tails from progress training.
- Select checkpoints using held-out episodes.
- Use `fail` primarily for auditing initially; later assess which prefixes and recovery segments are useful for BC.
- Compare against a model trained on the mixed collection.

Using known success labels for reference selection is an intentional departure from fully self-supervised SCIZOR. Success alone does not establish local quality, and selecting only the fastest successes could remove deliberate precision and recovery behavior.

### 4.2 Bins and sampling

Compare the released bins with boundaries **0, 1, 2, 4, 8, ∞ seconds**.

Avoid dividing every episode into five equal fractions: a long episode containing setbacks could normalize its own stall into an apparently ordinary interval. If phase-dependent timing becomes necessary, derive reference durations from clean successful behavior.

Adapt [the HDF5 sampler](../curation/suboptimal_classifier/dataset/hdf5_dataset.py) to:

1. Sample only feasible bins for the episode or valid segment.
2. Choose valid start/end pairs without clipping the endpoint and changing the intended class.
3. Balance episodes instead of weighting by frame count.
4. Initially cap sampling from the final category to **8–16 seconds**, while keeping the classification category open-ended.
5. Log realized class counts and time-gap distributions.

The final-bin sampling cap focuses training on useful temporal scales instead of easy, widely separated pairs. It should be validated, especially before introducing lookaheads longer than that training range.

### 4.3 Multiple inference horizons

Start with **2, 4, and 8 seconds**, with four seconds as the primary diagnostic horizon.

| Lookahead | Intended sensitivity | Main limitation |
|---|---|---|
| 2 s | Brief hesitation, missed contact, local pauses | May miss coherent-looking repeated retries |
| 4 s | Sustained stalls and failed attempts | Blurs attribution over several actions |
| 8 s | Repeated retries and back-and-forth behavior | Can hide failure followed by recovery |

Long failures do not automatically require equally long windows: repeated short-window evidence can identify a sustained stall. Longer windows help when local motions appear plausible but collectively accomplish little.

Preserve each horizon's scores. Calibrate separately before combining; the current raw scores are not directly comparable across horizons. Review localization and recovery preservation, not just whether an entire failure episode receives a high score.

### 4.4 Score construction and frequency dependence

The [existing scorer](../curation/video_encoding/suboptimal_hdf5.py) approximately computes:

```text
positive discrepancy between actual and predicted bin coordinates
    / number of frames in the window
    -> overlapping-window averaging
    -> backward discounted accumulation
```

Important caveats:

1. **Frequency effects partly cancel.** Dividing by frame count reduces the initial score at higher rates, but backward accumulation adds more terms per second. For a steady discrepancy away from boundaries, these effects approximately cancel. Do not scale thresholds simply by the frequency ratio.
2. **Long horizons can receive weaker scores.** Their frame-count denominator grows while the predicted bin index stays bounded.
3. **The final bin has little long-duration resolution.** Interpolation to a 10,000-second upper bound is not a meaningful physical model for these episodes.
4. Training and scoring use inconsistent bin-boundary conventions.
5. The backward recurrence skips index zero.
6. Scores are ordinal discrepancy measures, not probabilities of failure.

For a versioned new variant, consider the bounded ordinal deficit:

```text
d_h(t) = max(0, b(actual_elapsed_time) - sum(k * p_k for k in 0..4)) / 4
```

Here `b` must use exactly the same bin convention as training. Aggregate overlapping intervals with normalized weights and express normalized temporal smoothing in seconds. This remains an ordinal measure; it does not recover predicted seconds.

Start backward influence with a **one-second half-life**, then compare two seconds. The existing `gamma=0.5`, applied as `gamma**(1/freq)`, already has a one-second half-life. Longer failure windows do not automatically justify more distant backward blame.

These changes alter score calibration and require fresh thresholds. Preserve a separately named release-compatible baseline.

### 4.5 Local versus trajectory mixing

Start with `mix_level=0`; compare `0.1–0.25` later.

The released equal local/trajectory mixture can penalize useful sections because a distant part of the same episode is poor. This is particularly undesirable when preserving good prefixes and recoveries from episodes containing setbacks.

In this checkout, the policy's `mix_level` affects filtering. The scoring CLI's corresponding option affects visualization rather than the raw score saved to HDF5.

### 4.6 Operational definition of suboptimal behavior

| Behavior | Initial treatment |
|---|---|
| Sustained pause unrelated to contact or settling | Candidate for removal |
| Repeated unsuccessful retries without net progress | Candidate for removal |
| Clear retreat from the intended goal | Review; temporal distance may miss it |
| Deliberate alignment, force application, settling | Preserve unless shown unnecessary |
| Successful recovery | Preserve when possible |
| Stable terminal hold | Preserve a short deployment-relevant portion |
| Ambiguous or occluded behavior | Retain or downweight rather than confidently delete |

A temporal model can mistake large incorrect motion for progress. It cannot reliably distinguish productive contact from being stuck when relevant physical state is invisible.

Annotate approximately **20–30 varied episodes**, including failures, with behavior intervals. Select thresholds for high precision of removal. Treat 5%, 10%, and 20% removal as experimental budgets, not assumed contamination rates. Do not automatically transfer the paper's raw threshold or Robomimic deletion percentage.

## 5. Representation, optimization, and evaluation

The sampled top view provides overall context; wrist views reveal contact detail. At 84×84 the box and its edges occupy relatively few pixels.

| Setting | Initial recommendation |
|---|---|
| View | Top camera; audit failures with both wrist views |
| Resolution | 140×140; compare 224×224 for fine contact |
| Encoder | Frozen DINOv2 |
| Fusion blocks | 2, compared against released 6 |
| Batch size | 16 initially; measure memory before increasing |
| Learning rate | `1e-4` |
| Weight decay | `1e-3` |
| Initial budget | 3,000–5,000 updates with held-out selection |
| Validation interval | 500 updates |
| Sampling | Episode-balanced and feasible-bin-balanced |

Implementation issues in [the classifier](../curation/suboptimal_classifier/discriminator/discriminator.py):

- `RankHead` applies softmax before cross-entropy. A corrected variant should train on logits and expose probabilities for inference.
- The default branch computes feature differences but discards them. `cat_start_feature=True` enables start-plus-difference features; treat this as a separate architectural ablation.

With only 134 episodes, many adjacent frames do not imply many independent examples. Check overfitting by episode/session, per-bin accuracy, confusion, calibration, and annotated removal precision. High temporal-classification accuracy is not sufficient evidence of useful curation.

Frozen-feature caching is a useful later optimization because overlapping windows repeatedly encode the same frames. Multi-view fusion is another possible extension if a single view misses key state; it requires more than listing multiple camera keys in the current loader, whose model input is single-view.

## 6. Required implementation and adaptation

### 6.1 Raw-data adapter

The current HDF5 trainer cannot directly read the JPEG/PT layout. Add an adapter that:

1. Builds a manifest with episode ID, source group, date/session, timestamps, and verified outcome.
2. Constructs a regular progress grid from actual timestamps.
3. Selects images and records source indices and sampling error.
4. Aligns proprioception and action targets.
5. Writes derived HDF5 files and episode-level splits.
6. Preserves the mapping from progress times to policy times.

For the first single-view implementation, map the top camera to `agentview_image`:

```text
data/demo_0/
    obs/agentview_image   uint8 [T,H,W,3]
    actions               [T,14]
    timestamps            [T]
    attributes: num_samples, source_episode, session, outcome

mask/train
mask/valid
mask/progress_train
```

Timestamps and provenance are adapter additions. The existing progress loader ignores timestamps and assumes the configured regular frequency. This minimal progress schema is not necessarily the complete schema required by the selected policy implementation.

The progress model does not consume actions in this configuration, but the loader uses their length. Resolve policy actions correctly:

- Follower positions are observed state.
- Leader positions are candidate command targets, subject to the teleoperation mapping.
- Do not silently replace commanded actions with future follower state.
- Verify joint ordering, gripper conversion, command timing, and deployment conventions.
- Match policy observation alignment to execution-time availability; avoid supplying future camera frames.

Keep raw recordings unchanged. Use derived working files for score writers, which modify HDF5 in place.

### 6.2 Training and scoring additions

| Area | Required work |
|---|---|
| Configuration | ALOHA settings; matching sampler bins and classifier thresholds |
| Loss | Versioned logits-based cross-entropy with consistent inference probabilities |
| Pair sampling | Episode balancing, feasible intervals, class-distribution diagnostics |
| Evaluation | Held-out evaluation and checkpoint selection |
| Scoring | Versioned construction, per-horizon outputs, confidence, endpoint handling |
| Export | Scores/masks with episode IDs, timestamps, and provenance |

The existing scorer overwrites `subop_score` on repeated runs. Preserve distinct outputs for different horizons and scoring variants before conducting sweeps. Truncated lookaheads near episode ends require explicit handling; they should not be interpreted as full-horizon evidence.

### 6.3 Diffusion-policy integration

The bundled Robomimic fork has **no diffusion-policy implementation**. Its supplied templates train BC-GMM. Add score-aware sampling to the chosen diffusion-policy codebase.

The [current sequence loader](../robomimic/robomimic/utils/dataset.py) filters anchors but retrieves subsequent actions from original trajectories. A retained anchor can therefore supervise actions that were marked for rejection.

Initial conservative rule:

```text
Keep an anchor only if its entire non-padding supervised action chunk
contains no confidently rejected action.
```

Apply this to the exact action slice used by the diffusion loss, including history offsets. Preserve original time continuity. Never concatenate surviving frames into a synthetic continuous trajectory.

Report both rejected timesteps and rejected training anchors; chunk-level rejection can substantially amplify frame-level removal. Compare chunk weighting later if strict rejection discards too much useful data. Account explicitly for episode-boundary padding.

Express policy horizons in seconds: 16 actions span 0.8 seconds at 20 Hz but 0.32 seconds at 50 Hz.

## 7. Launching training

Use [the direct HDF5 trainer](../curation/suboptimal_classifier/train_hdf5.py), not the legacy trainer or distributed OXE launcher.

**The conversion output and masks below do not exist yet.** After conversion, this command is a compatibility baseline for the current implementation:

```bash
source scripts/activate_curation.sh

python -m curation.suboptimal_classifier.train_hdf5 \
  --name aloha_close_reference_seed42 \
  --config.hdf5_dataset_kwargs.data_dir=/absolute/path/derived/aloha_progress_10hz \
  --config.hdf5_dataset_kwargs.filter_key=progress_train \
  --config.hdf5_dataset_kwargs.freq=10 \
  --config.hdf5_dataset_kwargs.batch_size=16 \
  --config.hdf5_dataset_kwargs.num_workers=0 \
  --config.hdf5_dataset_kwargs.obs_keys.agentview_image=140 \
  --config.discriminator.num_blocks=2 \
  --config.num_steps=5000 \
  --config.save_interval=500
```

This still uses the **existing bins, sampler, and loss**, and does not add held-out evaluation. It is not the complete recommended variant.

For that variant, add an `aloha_close.py` configuration and select it with `--config=...`. Set both fields to the same new interval list:

```python
config.hdf5_dataset_kwargs.time_bins
config.discriminator.rank_thres
```

Sampler, loss, evaluation, and scoring changes require implementation beyond configuration.

Existing scoring, on a derived working copy:

```bash
python -m curation.video_encoding.suboptimal_hdf5 \
  --data-dir /absolute/path/derived/aloha_progress_10hz \
  --model-path runs/progress/aloha_close_reference_seed42/model_5000.pth \
  --goal-time 2 \
  --discount-gamma 0.5 \
  --batch-size 16 \
  --save-score
```

For the adapted model/scorer, use `--goal-time 2,4,8`. The existing CLI accepts that syntax but averages horizons using the current formula. Its frequency comes from the saved model configuration, so scoring data must match that sampling convention.

Choose a checkpoint using held-out results once evaluation is implemented; `model_5000.pth` above is only an example final checkpoint. The direct trainer currently has no automatic evaluation, early stopping, mixed precision, or optimizer-state resume.

An exact diffusion-policy launch command depends on the policy implementation and cannot be supplied from the Can/Square templates.

## 8. Recommended experiment sequence

1. **Resolve remaining deployment details.** Dataset group semantics and 50 Hz control are confirmed; verify command representation, gripper mapping, and timing.
2. **Convert and split data.** Use timestamps, preserve provenance, and hold out episodes/sessions before fitting or threshold selection.
3. **Establish policy baselines.** Compare `close` only, `close` plus all `fail`, and `close` plus locally curated `fail` to evaluate recovery preservation.
4. **Train a reference progress model.** Start with successful training episodes; compare mixed-data training separately.
5. **Audit predicted intervals.** Include stalls, deliberate contact, recovery, terminal holds, and low-scoring failures.
6. **Compare temporal choices.** Original bins/two-second inference versus wider bins and separate 2/4/8-second scores.
7. **Introduce conservative filtering.** Track removal precision, policy-chunk retention, and task-stage coverage.
8. **Evaluate policy outcomes.** Match update budgets, include random-removal controls, use multiple seeds, and measure real task success.
9. **Add deduplication after progress filtering works.** Avoid changing both components before their effects are understood.

For deduplication, raw absolute joint-position cosine similarity is a questionable behavior metric. Consider consistently scaled motion/action representations and preserve useful variation in contact and recovery. Keep any normalization fitted to training data only.

## 9. Main quality risks and decision points

- **Temporal change is not directional goal progress:** incorrect motion can look productive.
- **Frequent failure can become the learned norm:** compare successful-reference and mixed-data training.
- **Contact and occlusion create ambiguity:** inspect wrist views and preserve uncertain segments.
- **Long-window attribution can erase recovery:** retain per-horizon diagnostics and start with weak trajectory mixing.
- **Terminal holds can be falsely penalized:** distinguish completion from unproductive stalls.
- **Session appearance can leak group identity:** clean trajectories and trajectories containing recovered failures were collected on different dates; the folders are not terminal outcome labels.
- **Frame count exaggerates sample diversity:** split by episodes/sessions and monitor overfitting.
- **Policy chunks can contain rejected targets:** filter the supervised slice, not just the anchor.
- **Action or time alignment errors can dominate model quality:** verify these before hyperparameter tuning.
- **Thresholds are implementation-specific:** changing loss, bins, horizons, smoothing, or mixing requires recalibration.

The immediate priorities are timestamp alignment, verified action semantics, corrected and calibrated progress scoring, and diffusion action-chunk integration. Wider bins and longer horizons should be evaluated on top of those foundations.
