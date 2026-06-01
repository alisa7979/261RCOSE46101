# Sarcasm detection (Reddit and Twitter)

RoBERTa-based binary sarcasm classifier with Reddit context and emoji ablations, plus cross-domain evaluation between Reddit and Twitter.

## Project layout

- **preprocess.py** — Text cleaning, emoji removal, Reddit context formatting (`[CTX] parent [REPLY] reply`), label columns
- **train.py** — Train `roberta-base`; edit constants at the top to switch experiments
- **evaluate.py** — Test a saved checkpoint; in-domain and cross-domain test sets
- **scripts/collect_results.py** — Print metrics from `best_model_*/test_metrics.json`
- **results/experiment_summary.csv** — Final 6-experiment summary table
- **results/README.txt** — How the summary CSV was built

## Data

Place datasets next to this folder (paths are relative from `source_code_v2/`):

- **reddit_data** — `train.csv`, `val.csv`, `test.csv` with columns `reply`, `parent`, `label`
- **twitter_data** — `train.csv`, `val.csv`, `test.csv` with columns `tweet`, `sarcastic`

## Dependencies

```bash
pip install torch transformers datasets pandas scikit-learn emoji
```

## How to run

Run all commands from `source_code_v2/`.

### 1. Train

Edit the **Experiment config** block in `train.py`:

- `DATASET` — `"reddit"` or `"twitter"`
- `USE_CONTEXT` — `True` for Reddit context (ignored for Twitter)
- `REMOVE_EMOJI` — `True` to strip emojis

```bash
python train.py
```

Outputs:

- Checkpoint: `./best_model_<experiment_name>/`
- Training logs: `./results_<experiment_name>/`
- Validation metrics: `./best_model_<experiment_name>/metrics.json`

### 2. Evaluate

Edit the **Experiment config** block in `evaluate.py`:

- `EXPERIMENT_NAME` — must match the trained folder (e.g. `reddit_context_emoji`)
- `TEST_DATASET` — `"reddit"` or `"twitter"` (use the other domain for cross-domain eval)

Preprocessing settings (`remove_emoji`, `max_length`, etc.) are loaded from the checkpoint `metrics.json` when present.

```bash
python evaluate.py
```

Test metrics: `./best_model_<experiment_name>/test_metrics.json`

### 3. Collect printed results (optional)

```bash
python scripts/collect_results.py
```

## Experiment names (auto-generated)

- Reddit, no context, keep emoji → `reddit_no_context_emoji`
- Reddit, no context, no emoji → `reddit_no_context_no_emoji`
- Reddit, context, keep emoji → `reddit_context_emoji`
- Reddit, context, no emoji → `reddit_context_no_emoji`
- Twitter, no context, keep emoji → `twitter_no_context_emoji`
- Twitter, no context, no emoji → `twitter_no_context_no_emoji`

## Training settings

- Model: `roberta-base`, 2 labels
- 10 epochs; best checkpoint by validation F1 (`load_best_model_at_end=True`)
- `save_only_model=True` (weights only, no optimizer state in checkpoints)
- Max length: 128 (no context), 256 (Reddit with context)
- Class weights: `(1.0, 3.0)` for weighted cross-entropy

## Cross-domain evaluation

- Train on Reddit, test on Twitter: set `TEST_DATASET = "twitter"` in `evaluate.py`
- Train on Twitter, test on Reddit: set `TEST_DATASET = "reddit"`; context is disabled automatically for Twitter-trained models

See `results/experiment_summary.csv` for aggregated val and test F1 across all runs.
