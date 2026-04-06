# `classifer_training`

This folder is a clean, self-contained implementation for training a prompt-difficulty predictor from hidden states.

The core assumption is:

- the target label is **temperature-sampling accuracy per prompt**
- the main input is **hidden states**
- the feature space is flexible enough to also include **generation length, entropy, timing, or any other numeric metadata**

It is intentionally standalone. It does not depend on `CapBound`, although it can read CapBound-style hidden-state dumps and inference JSONL files.

## What is inside

- `aggregate_labels.py`
  - takes multiple sampled inference runs for the same prompt set
  - computes prompt-level `sampling_accuracy`
  - also aggregates optional numeric features like length, timing, and text entropy
- `prepare_datasets.py`
  - normalizes source datasets into a small JSONL format for this module
  - supports local DeepScaleR parquet files and Hugging Face datasets such as DAPO-Math-17k
- `extract_hidden_states.py`
  - extracts last-token prompt representations from a Hugging Face causal LM
  - supports `hidden`, `attn`, and `ffn` components
- `extract_rollout_hidden_states.py`
  - extracts one row per sampled rollout from `all_experiments.jsonl`
  - supports prompt-side hidden states plus response-side hidden states from the last answer/reasoning token
  - stores single-rollout numeric features in the index file so the existing trainer can use them directly
- `sample.py`
  - runs one temperature-sampled pass over a normalized dataset
  - writes `all_experiments.jsonl` and `evaluation_results.jsonl` in the exact format expected by label aggregation
- `make_manifest.py`
  - creates a manifest from a simple `<root>/<dataset>/<model_slug>/...` directory layout
- `train.py`
  - loads hidden states plus the aggregated labels
  - builds features from selected components and selected layers
  - trains either a regression model or a classifier
- `data.py`, `features.py`, `models.py`, `utils.py`
  - small helper modules so the pipeline stays easy to read

## Expected data

### 1. Hidden states

The training script supports a few simple formats:

- a CapBound-like `hiddenStates.pt`:
  - list of examples
  - each example is a dict like `{"ffn": [...], "attn": [...]}`
- a batched dict:
  - `{"task_ids": [...], "ffn": tensor[N, L, D], "attn": tensor[N, L, D]}`
- a plain tensor:
  - `tensor[N, L, D]`
  - this is treated as a single component named `hidden`

If the hidden-state file itself does not store `task_id`, pass an `index_path` that contains the example order and `task_id`.

This folder can generate such files directly with `extract_hidden_states.py`.

### 1a. Normalized datasets

`prepare_datasets.py` writes JSONL records like:

```json
{
  "dataset_name": "deepscaler",
  "task_id": "10720",
  "split": "train",
  "user_input": "question text ...",
  "ground_truth": "\\frac{15 \\sqrt{2}}{8}",
  "messages": [{"role": "user", "content": "question text ..."}]
}
```

This normalized file can be used both as:

- the input to `extract_hidden_states.py`
- the `index.jsonl` alignment file used later by `make_manifest.py`

### 2. Sampled runs

`aggregate_labels.py` expects each run directory to contain:

- `all_experiments.jsonl`
- `evaluation_results.jsonl`

This folder can generate those run directories directly with `sample.py`, where each run is one sampled pass over the same prompts.

### 3. Manifest

`train.py` uses a small JSON manifest:

```json
{
  "datasets": [
    {
      "name": "aime24",
      "hidden_states_path": "/abs/path/to/hiddenStates.pt",
      "index_path": "/abs/path/to/index.jsonl",
      "labels_path": "/abs/path/to/sampling_labels.jsonl"
    }
  ]
}
```

## Step 0: prepare datasets

DeepScaleR can be normalized from the local parquet files already present in this repo:

```bash
python -m classifer_training.prepare_datasets \
  --dataset_name deepscaler \
  --source auto
```

DAPO-Math-17k can be pulled from Hugging Face:

```bash
python -m classifer_training.prepare_datasets \
  --dataset_name dapo_math_17k \
  --source auto \
  --hf_dataset_id open-r1/DAPO-Math-17k-Processed \
  --hf_splits train \
  --train_examples 5000 \
  --validation_examples 500 \
  --test_examples 500
```

If you want both at once, edit or run:

- [prepare_target_datasets.sh](/home/jongwonlim/verl/yoonho/verl/classifer_training/examples/prepare_target_datasets.sh)

## Step 0.5: extract hidden states

After dataset normalization, extract prompt hidden states for `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`:

```bash
python -m classifer_training.extract_hidden_states \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --input_path classifer_training/artifacts/datasets/deepscaler \
  --dataset_name deepscaler \
  --components hidden \
  --hidden_root classifer_training/artifacts/hidden \
  --index_root classifer_training/artifacts/index
```

This writes:

- `hidden/<dataset>/<model_slug>/hidden_states.pt`
- `index/<dataset>/<model_slug>/index.jsonl`

If `input_path` is a directory, the extractor combines any existing `train.jsonl`, `validation.jsonl`, and `test.jsonl` files into one hidden-state artifact while preserving each example's `split` inside `index.jsonl`.

Ready-to-edit examples:

- [extract_deepscaler_hidden_states.sh](/home/jongwonlim/verl/yoonho/verl/classifer_training/examples/extract_deepscaler_hidden_states.sh)
- [extract_dapo_math_17k_hidden_states.sh](/home/jongwonlim/verl/yoonho/verl/classifer_training/examples/extract_dapo_math_17k_hidden_states.sh)

## Step 0.75: extract single-rollout hidden states

If you want the probe to see both the prompt and one sampled rollout, extract from an existing run
directory instead of the normalized prompt dataset:

```bash
python -m classifer_training.extract_rollout_hidden_states \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --run_dirs classifer_training/artifacts/runs/dapo_math_17k/deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B/temp0.7_seed1 \
  --dataset_name dapo_math_17k \
  --components prompt_hidden response_hidden \
  --layers 27 \
  --response_anchor reasoning_or_answer \
  --hidden_root classifer_training/artifacts/rollout_hidden \
  --index_root classifer_training/artifacts/rollout_index \
  --hidden_filename seed1_rollout_hidden_states.pt \
  --index_filename seed1_rollout_index.jsonl \
  --trust_remote_code
```

That writes:

- `rollout_hidden/<dataset>/<model_slug>/seed1_rollout_hidden_states.pt`
- `rollout_index/<dataset>/<model_slug>/seed1_rollout_index.jsonl`

Each row keeps the original prompt `task_id`, so prompt-level `difficulty` labels still align, but the
index file now also contains per-rollout numeric features under `rollout_features`.

## Step 1: aggregate prompt-level labels

First create multiple sampled runs over the same prompt set:

```bash
python -m classifer_training.sample \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --input_path classifer_training/artifacts/datasets/dapo_math_17k \
  --dataset_name dapo_math_17k \
  --output_dir classifer_training/artifacts/runs/dapo_math_17k/deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B/temp0.7_seed1 \
  --backend vllm \
  --grader math_verify \
  --temperature 0.7 \
  --top_p 1.0 \
  --seed 1 \
  --batch_size 64 \
  --trust_remote_code
```

Repeat the same command with different `--seed` values and output directories.

Then aggregate the run directories:

```bash
python -m classifer_training.aggregate_labels \
  --run_dirs \
    /path/to/temp07_seed1 \
    /path/to/temp07_seed2 \
    /path/to/temp07_seed3 \
  --output_path /path/to/sampling_labels.jsonl
```

This produces one row per prompt with:

- `sampling_accuracy`
- `difficulty = 1 - sampling_accuracy`
- `num_runs`
- aggregated feature statistics under `aggregated_features`

When using multiple hidden-state layers in `train.py`, the intended default interpretation is to
concatenate the selected layer vectors, not average them together. Use `--component_pooling concat`
unless you explicitly want pooled layer features.

Built-in aggregated features include:

- `input_length_*`
- `output_length_*`
- `generation_time_*`
- `think_tokens_*`
- `answer_tokens_*`
- `output_text_entropy_*`
- `reasoning_text_entropy_*`
- `answer_text_entropy_*`

You can also add arbitrary numeric fields from `all_experiments.jsonl`:

```bash
python -m classifer_training.aggregate_labels \
  --run_dirs /path/to/run_* \
  --run_glob "data/sampled/temp07_seed*/" \
  --extra_numeric_fields token_stats.total_tokens config.temperature \
  --output_path /path/to/sampling_labels.jsonl
```

## Qwen3-8B Reproduction

If you want to rerun the same prompt-level experiments on another server with a larger Qwen model,
the cleanest path is:

1. sample multiple temperature seeds on DAPO-Math-17k
2. aggregate prompt-level labels
3. optionally rebuild labels with official `Math-Verify`
4. train either the single-stage ET baseline or the two-stage variant

The scripts below assume:

- normalized DAPO dataset already exists under `classifer_training/artifacts/datasets/dapo_math_17k`
- the new server has a working Python environment for the main pipeline
- if you want official `Math-Verify`, use a Python `>=3.10` environment

### 1. Sample Qwen3-8B runs

Edit the model path if needed and run:

- [run_qwen3_8b_prompt_et.sh](/home/jongwonlim/verl/yoonho/verl/classifer_training/examples/run_qwen3_8b_prompt_et.sh)

This wraps the existing sampler and expects one run directory per seed.

### 2. Build official Math-Verify labels

If you want labels from the full generated trajectory instead of the local fallback scorer:

```bash
python -m classifer_training.make_mathverify_labels \
  --base_labels classifer_training/artifacts/labels/dapo_math_17k/qwen3_8b/sampling_labels_base.jsonl \
  --run_dirs classifer_training/artifacts/runs/dapo_math_17k/qwen3_8b/temp0.7_seed1 classifer_training/artifacts/runs/dapo_math_17k/qwen3_8b/temp0.7_seed2 \
  --output_path classifer_training/artifacts/labels/dapo_math_17k/qwen3_8b/sampling_labels_mathverify_fulltext.jsonl \
  --score_text fulltext
```

Or, if you specifically want to score only the extracted final answer:

```bash
python -m classifer_training.make_mathverify_labels \
  --base_labels classifer_training/artifacts/labels/dapo_math_17k/qwen3_8b/sampling_labels_base.jsonl \
  --run_dirs classifer_training/artifacts/runs/dapo_math_17k/qwen3_8b/temp0.7_seed1 classifer_training/artifacts/runs/dapo_math_17k/qwen3_8b/temp0.7_seed2 \
  --output_path classifer_training/artifacts/labels/dapo_math_17k/qwen3_8b/sampling_labels_mathverify_answer.jsonl \
  --score_text answer
```

`make_mathverify_labels.py` preserves the existing aggregated features from the base label file and only
replaces the correctness-derived fields such as `sampling_accuracy`, `difficulty`, and run counts.

### 3. Train the prompt-level ET baselines

The reusable trainer is:

- [train_prompt_et.py](/home/jongwonlim/verl/yoonho/verl/classifer_training/train_prompt_et.py)

Single-stage ET:

```bash
python -m classifer_training.train_prompt_et \
  --manifest classifer_training/artifacts/manifests/dapo_math_17k_qwen3_8b.json \
  --labels_path classifer_training/artifacts/labels/dapo_math_17k/qwen3_8b/sampling_labels_enriched_mathverify_fulltext.jsonl \
  --output_dir classifer_training/artifacts/models/dapo_math_17k_qwen3_8b_single_stage_et \
  --mode single_stage \
  --n_estimators 2000 \
  --min_samples_leaf 5 \
  --max_features 0.5
```

Two-stage ET:

```bash
python -m classifer_training.train_prompt_et \
  --manifest classifer_training/artifacts/manifests/dapo_math_17k_qwen3_8b.json \
  --labels_path classifer_training/artifacts/labels/dapo_math_17k/qwen3_8b/sampling_labels_enriched_mathverify_fulltext.jsonl \
  --output_dir classifer_training/artifacts/models/dapo_math_17k_qwen3_8b_two_stage_et \
  --mode two_stage \
  --n_estimators 2000 \
  --min_samples_leaf 5 \
  --max_features 0.5
```

Both modes write:

- `summary.json`
- `predictions_test.jsonl`
- `prediction_alignment.png`

### Recommended comparison order

For the Qwen3-8B server run, compare these in order:

1. single-stage ET with the original aggregated labels
2. single-stage ET with official `Math-Verify` full-trajectory labels
3. two-stage ET with the same feature set and the same `Math-Verify` labels

That isolates:

- model-size effects
- scorer/label effects
- single-stage vs two-stage effects

## Step 2: train a regressor

Predict prompt-level sampling accuracy directly:

```bash
python -m classifer_training.train \
  --manifest /path/to/manifest.json \
  --output_dir /path/to/out/regression \
  --task_type regression \
  --target_field sampling_accuracy \
  --model ridge \
  --components ffn attn \
  --layers 12,13,14,15 \
  --component_pooling concat \
  --extra_features \
    label.aggregated_features.output_length_mean \
    label.aggregated_features.output_text_entropy_mean
```

## Manifest helper

If you keep artifacts under a simple directory convention:

- `hidden_root/<dataset>/<model_slug>/hidden_states.pt`
- `index_root/<dataset>/<model_slug>/index.jsonl`
- `labels_root/<dataset>/<model_slug>/sampling_labels.jsonl`

you can generate the manifest automatically:

```bash
python -m classifer_training.make_manifest \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --datasets deepscaler dapo_math_17k \
  --hidden_root /abs/path/to/hidden \
  --index_root /abs/path/to/index \
  --labels_root /abs/path/to/labels \
  --index_filename index.jsonl \
  --output_path /abs/path/to/manifest.json
```

`model_slug` is produced automatically from the model name.

## Concrete target setup

For your current setup, the intended pair is:

- model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- datasets:
  - `deepscaler` for DeepScaleR
  - `dapo_math_17k` for DAPO-math 17K

This repo already contains:

- local DeepScaleR parquet files under [data/deepscaler](/home/jongwonlim/verl/yoonho/verl/data/deepscaler)
- DAPO should now use the processed Hugging Face dataset `open-r1/DAPO-Math-17k-Processed`, which currently exposes about `17.4k` train rows instead of the oversized `1.79M` source artifact

Ready-to-edit example launchers are here:

- [prepare_target_datasets.sh](/home/jongwonlim/verl/yoonho/verl/classifer_training/examples/prepare_target_datasets.sh)
- [extract_deepscaler_hidden_states.sh](/home/jongwonlim/verl/yoonho/verl/classifer_training/examples/extract_deepscaler_hidden_states.sh)
- [extract_dapo_math_17k_hidden_states.sh](/home/jongwonlim/verl/yoonho/verl/classifer_training/examples/extract_dapo_math_17k_hidden_states.sh)
- [run_deepscaler_regression.sh](/home/jongwonlim/verl/yoonho/verl/classifer_training/examples/run_deepscaler_regression.sh)
- [run_dapo_math_17k_classification.sh](/home/jongwonlim/verl/yoonho/verl/classifer_training/examples/run_dapo_math_17k_classification.sh)

## Step 3: train a classifier

Predict whether a prompt is easy or hard:

```bash
python -m classifer_training.train \
  --manifest /path/to/manifest.json \
  --output_dir /path/to/out/classification \
  --task_type classification \
  --target_field sampling_accuracy \
  --classification_threshold 0.5 \
  --train_splits train \
  --eval_splits validation \
  --test_splits test \
  --model logistic \
  --components ffn \
  --layers 14:18 \
  --component_pooling mean \
  --extra_features label.aggregated_features.output_length_mean
```

## Supported model families

Regression:

- `linear`
- `ridge`
- `mlp`
- `random_forest`

Classification:

- `logistic`
- `linear_svm`
- `mlp`
- `random_forest`

## Outputs

Each training run writes:

- `model.joblib`
- `metrics.json`
- `feature_spec.json`
- `predictions.jsonl`

When `--train_splits` / `--eval_splits` / `--test_splits` are provided, `train.py` trains on the explicit split names found in `index.jsonl` instead of doing a random `train_test_split`.

## Local requirements

This folder is meant to stay self-contained. If needed, install its extra dependency set separately:

```bash
python -m pip install -r classifer_training/requirements.txt
```
