# classifer_training

This folder is for one line of work only:

- input: a single trajectory
- target: prompt-level `value = 1 - difficulty`
- model family: Ridge on prompt hidden, response hidden, and optional rollout scalars

If code or results do not belong to that line, they go in `deprecated/`.

## Active workflow

There are two active paths.

1. Weak training on DAPO weak4
- build prompt-level labels
- extract prompt hidden states
- extract response hidden states and rollout index features
- train a Ridge value model

2. Transfer evaluation on another dataset or another Qwen3-4B checkpoint
- generate rollouts
- score each trajectory if a `reward` feature is needed
- extract prompt/response hidden states
- run the saved model on the new artifacts

## Main scripts

Core weak-training scripts:

- `build_weak_prompt_dataset_and_labels.py`
- `prepare_weak4_shards.py`
- `extract_hidden_states.py`
- `extract_rollout_hidden_states.py`
- `train_weak_only_single_rollout_hidden.py`
- `eval_single_rollout_hidden_transfer.py`

Supporting utilities:

- `single_rollout_hidden_utils.py`
- `rollout_utils.py`

Transfer helpers:

- `prepare_ifbench_dataset.py`
- `rescore_ifbench_run.py`

Current generation entrypoint is `sample.py`.

## Recommended entrypoints

Weak end-to-end rerun:

- `examples/run_weak_single_trajectory_e2e.sh`

Transfer chains:

- `examples/run_math_transfer_gpu_chain.sh`
- `examples/run_ifbench_transfer_gpu_chain.sh`
- `examples/run_transfer_gpu_chain.sh`
- `examples/run_dapo_weak4_trained_transfer_gpu_chain.sh`
- `examples/run_dapo_trained_transfer_gpu_chain.sh`

## Model contract

The active target is always prompt-level `value`.

One prompt can have multiple sampled trajectories.
Each trajectory becomes one row, but all rows from the same prompt share the same target value.

If `reward` is included as a feature, it means per-trajectory correctness (`0/1`) and is only a feature.
It is not the training target.

## PCA contract

If PCA is enabled, it is part of the trained model state.

That means:

- PCA is fit on the training split only
- the fitted PCA objects are saved inside `model.joblib`
- transfer evaluation must reuse those saved PCA objects
- transfer code must not refit PCA on the target dataset

Older bundle-patching helpers were moved to `deprecated/`.

## Weak training

For a full rerun:

```bash
GPU_IDS="0 1 2 3" \
ROLLOUT_COMPONENT="think_end_hidden" \
PROMPT_HIDDEN_PCA_DIM=0 \
ROLLOUT_HIDDEN_PCA_DIM=32 \
bash classifer_training/examples/run_weak_single_trajectory_e2e.sh
```

That script does:

1. weak labels
2. weak prompt shards
3. prompt hidden extraction
4. response hidden extraction and rollout-index writing
5. optional clean feature collection
6. weak-only Ridge training

If step 4 is interrupted, rerun the same command with `OVERWRITE=0`.
The response extractor resumes from shard checkpoints.

## DeepScaleR custom pipeline

Main entrypoint:

- `run_deepscaler_custom_qwen3_4b_base_4gpu.sh`

This script now supports two generation modes:

- `GENERATION_PARALLELISM=tp`
  - one vLLM worker uses all visible GPUs together
  - `TP_SIZE` defaults to the number of visible GPUs
  - use this when the model does not fit on one GPU
- `GENERATION_PARALLELISM=shard`
  - one vLLM worker per GPU
  - generation shards run in parallel in waves
  - `TP_SIZE` must stay `1`
  - use this when the model fits on one GPU and you want faster wall-clock generation

`GPU_IDS` accepts either `0,1,2,3` or `0 1 2 3`.

Single-GPU run:

```bash
PYTHON=/home/jongwonlim/miniconda/envs/jongwon/bin/python \
GPU_IDS=0 \
TRAIN_PROMPTS=4096 \
VALIDATION_PROMPTS=1024 \
TRAIN_NUM_SAMPLES=2 \
VALIDATION_NUM_SAMPLES=16 \
LAYERS=19 \
PROMPT_LAST_N_VALUES=10 \
ROLLOUT_COMPONENTS=response_last10_mean_hidden \
bash classifer_training/run_deepscaler_custom_qwen3_4b_base_4gpu.sh
```

Multi-GPU tensor-parallel generation:

```bash
PYTHON=/home/jongwonlim/miniconda/envs/jongwon/bin/python \
GPU_IDS=0,1,2,3 \
GENERATION_PARALLELISM=tp \
TRAIN_PROMPTS=4096 \
VALIDATION_PROMPTS=1024 \
TRAIN_NUM_SAMPLES=2 \
VALIDATION_NUM_SAMPLES=16 \
LAYERS=19 \
PROMPT_LAST_N_VALUES=10 \
ROLLOUT_COMPONENTS=response_last10_mean_hidden \
bash classifer_training/run_deepscaler_custom_qwen3_4b_base_4gpu.sh
```

Multi-GPU shard-parallel generation:

```bash
PYTHON=/home/jongwonlim/miniconda/envs/jongwon/bin/python \
GPU_IDS=0,1,2,3 \
GENERATION_PARALLELISM=shard \
TRAIN_PROMPTS=4096 \
VALIDATION_PROMPTS=1024 \
TRAIN_NUM_SAMPLES=2 \
VALIDATION_NUM_SAMPLES=16 \
LAYERS=19 \
PROMPT_LAST_N_VALUES=10 \
ROLLOUT_COMPONENTS=response_last10_mean_hidden \
bash classifer_training/run_deepscaler_custom_qwen3_4b_base_4gpu.sh
```

If you already finished the `4096/2048` train generation and only want a new validation split, reuse the old train run:

```bash
PYTHON=/home/jongwonlim/miniconda/envs/jongwon/bin/python \
GPU_IDS=0,1,2,3 \
GENERATION_PARALLELISM=shard \
TRAIN_PROMPTS=4096 \
VALIDATION_PROMPTS=1024 \
TRAIN_NUM_SAMPLES=2 \
VALIDATION_NUM_SAMPLES=16 \
LAYERS=19 \
PROMPT_LAST_N_VALUES=10 \
ROLLOUT_COMPONENTS=response_last10_mean_hidden \
REUSE_TRAIN_FROM_VALIDATION_PROMPTS=2048 \
bash classifer_training/run_deepscaler_custom_qwen3_4b_base_4gpu.sh
```

Notes:

- `GENERATION_PARALLELISM=shard` only changes the generation stage.
- prompt hidden extraction and rollout hidden extraction were already sharded by `NUM_SHARDS`.
- rerunning the same command with `OVERWRITE=0` still reuses finished shard outputs.

## Transfer evaluation

`eval_single_rollout_hidden_transfer.py` evaluates an already-trained bundle on another dataset.

Expected artifacts:

- prompt hidden tensors
- prompt hidden index files
- response hidden tensors
- response rollout index files
- prompt-level labels

Outputs per split:

- `predictions_<split>.jsonl`
- `prediction_diagnostics_<split>.png`
- `summary.json`

## What belongs where

`artifacts/`
- local results only
- hidden states, rollout indexes, labels, plots, models

`external/`
- local clones such as IFBench
- not meant to be committed

`deprecated/`
- old experiments
- old datasets
- old scripts
- old model families

## What is intentionally ignored

The repository ignores heavy local-only content under:

- `classifer_training/artifacts/`
- `classifer_training/external/`
- checkpoint directories
- cached tensors such as `*.pt`

This folder should stay light in git. Large generated outputs should stay under ignored paths only.

## Moved out of active tree

These are no longer treated as active entrypoints:

- one-off extraction helpers
- old sampling-coupled shell scripts
- old JSONL-to-run conversion helpers

They were moved to `deprecated/examples/` or `deprecated/code/`.
