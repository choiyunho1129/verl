# Probe Status

Last updated: 2026-04-22

This note reflects the current source-setting result before the transfer
experiments. The current recommendation is no longer a split prompt/trajectory
head, score stack, gated combiner, or MLP. The documented model family is a
single ridge regression over prompt hidden state, trajectory hidden state, and
entropy features.

## Current Setting

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Data: DAPO-Math-17K English subset through the `spo_temp1_subset0to4` artifact
- Split:
  - Train: shards/runs `0,1`
  - Validation: shards/runs `2,3,4`
- Label:
  - For each prompt, generate 16 rollouts.
  - `sampling_accuracy = correct_count / 16`
  - Probe difficulty is `1 - predicted_sampling_accuracy`

The probe is trained to predict `sampling_accuracy`. Difficulty is produced
only at the end:

```text
predicted_difficulty = 1 - predicted_sampling_accuracy
```

## Recommended Probe Family

### Input

Use only:

```text
x = [prompt_hidden_pca, trajectory_hidden_pca, entropy_features]
```

Prompt hidden:

- Extraction: `classifer_training.extract_hidden_states`
- Token pooling: `lastn_mean`
- Last tokens: `last_n = 6`
- Layer: `26`
- Raw hidden dim: `2560`
- PCA dim: `128`

Trajectory hidden:

- Extraction: `classifer_training.extract_rollout_hidden_states`
- Component: `response_hidden`
- Extracted layer: `26`
- Pooling: `mean`
- Raw hidden dim: `2560`
- PCA dim: `128`

Entropy features:

- Direct rollout entropy fields:
  - `output_mean_token_entropy`
  - `reasoning_mean_token_entropy`
  - `output_last_token_entropy`
  - `output_max_token_entropy`
  - `output_min_token_entropy`
  - `reasoning_last_token_entropy`
  - `reasoning_max_token_entropy`
  - `reasoning_min_token_entropy`
  - `answer_last_token_entropy`
  - `answer_max_token_entropy`
  - `answer_mean_token_entropy`
  - `answer_min_token_entropy`
- Derived entropy fields:
  - `entropy_gap_reasoning_answer`
  - `answer_entropy_gap_vs_output`
- Extra nested entropy field:
  - `rollout_features.answer_mean_token_entropy`

No prompt scalar features are used. No length, repetition, boxed-answer,
completion, logprob, margin, or token-position rollout scalars are used in the
recommended model.

Prompt-level optimized feature dimension:

```text
128 prompt PCA + 128 trajectory PCA + 15 entropy scalars = 271 dims
```

### Model

The model is a single ridge regression:

```text
f(x) = w^T StandardScaler(x) + b
```

Loss:

```text
L(w, b) = sum_i (y_i - f(x_i))^2 + alpha * ||w||_2^2
```

Best prompt-level hyperparameter in the current sweep:

```text
alpha = 10
```

Prompt-level optimized training target:

```text
y = prompt-level rollout16 sampling_accuracy
```

Row-level optimized training target:

```text
y = other_rollout_correctness
```

The row-level optimized model still uses a single ridge regression. It does not
create separate prompt and trajectory heads. The only change is the scalar
target used for training rows: each trajectory row predicts the correctness of
the other sampled train trajectory for the same prompt.

Prompt-level inference:

```text
predicted_sampling_accuracy(prompt)
  = mean over selected trajectory-row predictions
```

For the validation result below, all 16 validation trajectories are averaged.
If only 2 trajectories are available at inference time, average the 2 available
row predictions with the same estimator.

## Source Validation Results

Validation set:

- Prompts: `2115`
- Rows: `33840`
- Validation rows per prompt: `16`
- Train rows: `11294`

Headline metric is prompt-level performance after averaging row predictions per
prompt. Row-level metrics are secondary because all 16 rows for a prompt share
the same prompt-level label.

| Variant | Target | Layer | Prompt PCA | Rollout PCA | Dim | Model | Prompt R2 | Prompt MAE | Prompt RMSE | Pearson | Spearman | Row R2 | Row MAE |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| support-default single ridge | prompt mean | 26 | 16 | 16 | 52 | `ridge_a300` | 0.9350 | 0.0649 | 0.1054 | 0.9729 | 0.9244 | 0.7588 | 0.1161 |
| prompt-level optimized single ridge + entropy | prompt mean | 26 | 128 | 128 | 271 | `ridge_a10` | 0.9326 | 0.0668 | 0.1074 | 0.9721 | 0.9223 | 0.7428 | 0.1215 |
| row-R2 optimized single ridge + entropy | other rollout | 26 | 32 | 256 | 303 | `ridge_a0.01` | 0.8914 | 0.1016 | 0.1363 | 0.9596 | 0.9160 | 0.7800 | 0.1305 |
| single ridge + entropy | prompt mean | 26 | 256 | 256 | 527 | `ridge_a10` | 0.9325 | 0.0686 | 0.1075 | 0.9723 | 0.9218 | 0.7491 | 0.1217 |
| single ridge + entropy | prompt mean | 26 | 512 | 512 | 1039 | `ridge_a10` | 0.9273 | 0.0736 | 0.1116 | 0.9706 | 0.9181 | 0.7516 | 0.1240 |
| single ridge + entropy | prompt mean | 19 | 256 | 256 | 527 | `ridge_a10` | 0.9325 | 0.0685 | 0.1075 | 0.9722 | 0.9219 | 0.7505 | 0.1215 |
| older split score-stack, all features | split heads | 19 | 0 | 24 | 2666 | `pa10000_ta30_ca1000` | 0.9002 | 0.0871 | 0.1307 | 0.9554 | 0.9130 | 0.8075 | 0.1127 |
| older single ridge, all features | prompt mean | 26 | 512 | 512 | 1106 | `ridge_a10000` | 0.9083 | 0.0872 | 0.1252 | 0.9639 | 0.9183 | 0.7783 | 0.1218 |

Conclusion:

- The support-default simple ridge model is currently best for final prompt
  difficulty prediction among the runs in this note.
- The row-R2 optimized simple ridge model raises row R2 from `0.7428` to
  `0.7800` by using `other_rollout_correctness`, prompt PCA 32, rollout PCA
  256, and `alpha=0.01`.
- Adding many hand-built rollout scalars is not needed for the current
  documentable simple model.
- The old split-head score-stack has better row R2, but worse prompt-level R2.
  Since the final probe output is prompt difficulty, prompt-level performance
  should remain the headline number unless the analysis specifically targets
  row-level behavior.

## Visualizations

Prompt-level GT vs prediction:

![Prompt scatter](artifacts/probe/spo_temp1_subset0to4_simple_ridge_prompt_traj_entropy_summary/best_prompt_pred_vs_gt_scatter.png)

Prompt-level sorted alignment:

![Sorted prompt alignment](artifacts/probe/spo_temp1_subset0to4_simple_ridge_prompt_traj_entropy_summary/best_prompt_sorted_alignment.png)

Row-level density plot:

![Row hexbin](artifacts/probe/spo_temp1_subset0to4_simple_ridge_prompt_traj_entropy_summary/best_row_pred_vs_gt_hexbin.png)

## Artifacts

Best model directory:

```text
classifer_training/artifacts/probe/spo_temp1_subset0to4_simple_ridge_prompt_traj_entropy_layer26_pca128_promptmean
```

Important files:

```text
model.joblib
estimator_config.json
summary.json
predictions_weakval.jsonl
predictions_weakval_rows.jsonl
prediction_diagnostics_weakval.png
prediction_diagnostics_weakval_rows.png
```

Summary and visualization directory:

```text
classifer_training/artifacts/probe/spo_temp1_subset0to4_simple_ridge_prompt_traj_entropy_summary
```

Important files:

```text
metrics_summary.md
metrics_summary.json
best_prompt_pred_vs_gt_scatter.png
best_prompt_sorted_alignment.png
best_row_pred_vs_gt_hexbin.png
```

Row-R2 optimized model directory:

```text
classifer_training/artifacts/probe/spo_temp1_subset0to4_simple_ridge_entropy_rowr2_best_other_p32_r256
```

Row-R2 sweep directory:

```text
classifer_training/artifacts/probe/spo_temp1_subset0to4_simple_ridge_entropy_rowr2_sweep
```

Support-default model directory:

```text
classifer_training/artifacts/probe/spo_temp1_subset0to4_support_default_setting_single_ridge
```

Support-default visualization directory:

```text
classifer_training/artifacts/probe/spo_temp1_subset0to4_support_default_setting_visualization
```

## Transfer Notes

Transfer experiments to other datasets and trained checkpoints are still useful,
but they are not the current headline result for this document. Earlier
transfer work suggested that large full-hidden/full-scalar feature spaces can
overfit when the target split has few independent prompts. For transfer, use
the same principle as above:

```text
simple model + PCA-compressed hidden states + minimal scalar features
```

Do not use the old full 2666-dim split score-stack as the default transfer
probe unless the target validation result specifically justifies it.
