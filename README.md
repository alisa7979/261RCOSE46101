# Multimodal Sarcasm Detection

## Overview
This project implements sarcasm detection using RoBERTa-based transformer models trained on Reddit and Twitter datasets. The project investigates the effects of conversational context, emoji information, and cross-domain generalization across social media platforms. Developed as part of COSE461 Natural Language Processing at Korea University.

## Repository Structure

| Folder | Description |
|---|---|
| `source_code_v2/` | Preprocessing, training, evaluation, and experiment scripts |
| `reddit_data/` | Reddit dataset splits (train / val / test) |
| `twitter_data/` | Twitter dataset splits (train / val / test) |

> **Note:** Large dataset files (>100MB) are excluded from this repository due to GitHub file size limits.

## Experiments

The project evaluates:

- Reddit vs Twitter sarcasm detection
- Context vs no-context inputs
- Emoji ablation
- Cross-domain evaluation (Reddit → Twitter and Twitter → Reddit)

Experiment results are summarized in:

`source_code_v2/results/experiment_summary.csv`

## Requirements

```bash
pip install torch transformers datasets scikit-learn pandas numpy
```

## Usage

```bash
# Preprocess data
python source_code_v2/preprocess.py

# Train model
python source_code_v2/train.py

# Evaluate
python source_code_v2/evaluate.py
```

## Team
Korea University — COSE461 Natural Language Processing, Spring 2026  
Team 23
