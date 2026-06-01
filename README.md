# Multimodal Sarcasm Detection

## Overview
This project implements sarcasm detection using transformer-based models (RoBERTa, CLIP) trained on Reddit and Twitter datasets. Developed as part of COSE461 Natural Language Processing at Korea University.

## Repository Structure

| Folder | Description |
|---|---|
| `source_code/` | Baseline model — preprocessing, training, evaluation |
| `source_code_v2/` | Improved model with additional experiments |
| `reddit_data/` | Reddit dataset splits (train / val / test) |
| `twitter_data/` | Twitter dataset splits (train / val / test) |

> **Note:** Large dataset files (>100MB) are excluded from this repository due to GitHub file size limits.

## Requirements

```bash
pip install torch transformers datasets scikit-learn pandas numpy
```

## Usage

```bash
# Preprocess data
python source_code/preprocess.py

# Train model
python source_code/train.py

# Evaluate
python source_code/evaluate.py
```

## Team
Korea University — COSE461 Natural Language Processing, Spring 2025  
Team 23
