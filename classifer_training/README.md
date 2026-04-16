# classifer_training

This folder now only keeps the weak single-trajectory PCA + Ridge pipeline.

If a script or result does not belong to that pipeline, it has been moved to `deprecated/`.

## What is kept

Active scripts:

- `build_weak_prompt_dataset_and_labels.py`
  - build weak prompt splits and weak prompt-level labels from existing sampled run directories
- `prepare_weak4_shards.py`
  - shard the weak prompt dataset for hidden-state extraction
- `extract_hidden_states.py`
  - extract prompt hidden states from the weak prompt dataset
- `extract_rollout_hidden_states.py`
  - extract response-side hidden states from sampled weak rollouts
- `enrich_rollout_index.py`
  - add rollout scalar features, including actual token-entropy summaries, to an index JSONL
- `train_weak_only_single_rollout_hidden.py`
  - current training script for the weak single-trajectory Ridge models

Small helpers that are still used by the active pipeline:

- `data.py`
- `rollout_utils.py`
- `utils.py`

## Data layout

Only the weak prompt datasets stay in `artifacts/datasets/`:

- `dapo_math_17k_weak4`
- `dapo_math_17k_weak4_shards`
- `dapo_math_17k_weak4_v2`
- `dapo_math_17k_weak4_val20`

Older datasets were moved to `deprecated/artifacts/datasets/`.

## Current model results kept here

Only weak single-trajectory model results stay in `artifacts/models/`.
Everything else was moved to `deprecated/artifacts/models/`.

The main current search directories are:

- `weak4_val20_feature_growth_search`
- `weak4_val20_feature_ablation_pca16`
- `weak4_val20_feature_ablation_pca16_fast`
- `weak4_val20_prompt_hidden_small_search`
- `weak4_val20_prompt_plus_thinkend_small_search`

## Minimal pipeline

1. Build the weak prompt dataset and weak labels.
2. Extract prompt hidden states from the weak prompt dataset.
3. Extract response-side hidden states from sampled weak rollouts.
4. Enrich the rollout index with scalar rollout features.
5. Train the single-trajectory Ridge model.

## Example scripts

The active examples are:

- `examples/run_extract_dapo_math_17k_weak4_hidden_4gpu.sh`
- `examples/run_extract_dapo_math_17k_weak4_hidden_one.sh`
- `examples/run_single_trajectory_actual_entropy_refresh.sh`
- `examples/run_response_hidden_extract_after_sampling.sh`

Other old example scripts were moved to `deprecated/examples/`.

## Notes

- `train_weak_only_single_rollout_hidden.py` is now self-contained. It no longer imports helper functions from the old two-rollout or prompt-only experiment scripts.
- `deprecated/` is for old code, old datasets, and old results that we are not actively maintaining.
- New work in this folder should stay inside the weak single-trajectory PCA + Ridge line.
