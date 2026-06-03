# SAE Transfer

This project now includes the same SAE-style transfer mechanism used by the
previous ARC/MATH experiment.

## Stages

```text
tasks.jsonl
  ↓
run-sweep
  writes raw attempts, summary.csv, solved_ids.txt
  ↓
extract-sae
  writes sparse task-layer SAE feature rows
  ↓
build-sae-neighbors
  solved anchors from summary + unsolved targets from summary
  optional target restriction from curriculum plan
  ↓
run-sae-transfer
  top-1 anchor prompt + target prompt
```

## Separation From Knapsack

The curriculum/knapsack plan controls which target tasks are admitted to a run.
It does not change anchor selection. Anchor selection is always by SAE neighbor
score among solved anchors.

Use:

```bash
gct build-sae-neighbors \
  --summary runs/competitive_math/sweep_summary.csv \
  --target-plan runs/competitive_math/plans/curriculum.jsonl
```

If `--target-plan` is omitted, all unsolved targets in the summary are used.

## Feature Rows

`extract-sae` writes JSONL rows:

```json
{
  "task_id": "task_...",
  "layer": 10,
  "hookpoint": "layers.10",
  "features": {"L10:123": 0.42},
  "mean_l0": 38.1,
  "n_tokens": 221,
  "metadata": {"level": "2", "category": "Algebra"}
}
```

## Neighbor Scores

Supported variants:

```text
cos07        0.70 cosine + 0.15 weighted_jaccard + 0.15 tanimoto
cos01        0.10 cosine + 0.45 weighted_jaccard + 0.45 tanimoto
arc_weighted 0.70 SAE cosine + 0.20 weighted_jaccard + 0.10 structure_cosine
```

`structure_cosine` is metadata cosine over level/category by default. This is
formula-compatible with the ARC-weighted run, but ARC's original grid-structure
features are domain-specific and are not reused for non-grid datasets.

## Required Inputs

`run-sweep` produces the summary file used by `build-sae-neighbors`. CSV and
JSONL summaries are accepted. Each row must contain:

```text
task_id
solved
```

Optional metadata:

```text
level
category
metadata
```

`run-sae-transfer` needs `--anchor-solutions`, usually
`runs/<name>/sweep/raw.jsonl`. It uses the first exact successful raw output per
solved anchor.

## Generic Sweep Grading

`run-sweep` extracts answers from boxed output, `####`-style output, common
"final answer" phrases, or the last numeric token. It then compares against the
dataset answer using normalized string match and numeric/fraction match.

This is enough for Competitive MATH/GSM8K-style tasks. For datasets with custom
grading semantics, add a grader module and call it from `gct/sweep/runner.py`.
