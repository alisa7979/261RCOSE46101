"""Evaluate a trained checkpoint on a test split. Loads settings from metrics.json."""

import os
import json
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from preprocess import DATA_PATHS, default_max_length, preprocess_dataframe


# Experiment config
# Change these constants to switch runs.
EXPERIMENT_NAME = "reddit_context_emoji"  # Must match trained checkpoint folder name

# Test split: "reddit" or "twitter" (can differ from training for cross-domain)
TEST_DATASET = "twitter"

# Fallbacks if metrics.json is missing from the checkpoint
USE_CONTEXT = False
REMOVE_EMOJI = False
MAX_LENGTH = 128

MODEL_PATH = f"./best_model_{EXPERIMENT_NAME}"
TEST_PATH = DATA_PATHS[TEST_DATASET]["test"]


def load_train_config(model_path):
    config_path = os.path.join(model_path, "metrics.json")
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r") as f:
        return json.load(f)


def resolve_eval_settings(train_config):
    """Match preprocessing to training; apply cross-domain context rules."""
    train_dataset = train_config.get("dataset")
    use_context = train_config.get("use_context", USE_CONTEXT)
    remove_emoji = train_config.get("remove_emoji", REMOVE_EMOJI)
    max_length = train_config.get("max_length", MAX_LENGTH)

    # Context only applies on Reddit test rows.
    use_context = use_context and TEST_DATASET == "reddit"

    # Models trained on Twitter never saw [CTX] tokens.
    if train_dataset == "twitter":
        use_context = False

    if not max_length:
        max_length = default_max_length(use_context)

    cross_domain = train_dataset is not None and train_dataset != TEST_DATASET

    return use_context, remove_emoji, max_length, train_dataset, cross_domain


def build_dataset(df, tokenizer, use_context, remove_emoji, max_length):
    """Preprocess and tokenize the test split."""
    df = preprocess_dataframe(
        df,
        use_context=use_context,
        remove_emoji=remove_emoji,
    )

    dataset = Dataset.from_pandas(df[["text", "labels"]])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return df, dataset


def save_metrics(metrics):
    """Save test metrics inside the model folder."""
    output_path = os.path.join(MODEL_PATH, "test_metrics.json")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)


# Load checkpoint
train_config = load_train_config(MODEL_PATH)
use_context, remove_emoji, max_length, train_dataset, cross_domain = resolve_eval_settings(
    train_config
)

test_df = pd.read_csv(TEST_PATH, encoding="utf-8")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Run test evaluation
test_df, test_dataset = build_dataset(
    test_df, tokenizer, use_context, remove_emoji, max_length
)

trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset)

preds = torch.argmax(torch.tensor(predictions.predictions), dim=-1).numpy()
labels = test_df["labels"].values

metrics = {
    "experiment": EXPERIMENT_NAME,
    "train_dataset": train_dataset,
    "test_dataset": TEST_DATASET,
    "cross_domain": cross_domain,
    "use_context": use_context,
    "remove_emoji": remove_emoji,
    "max_length": max_length,
    "test_f1": f1_score(labels, preds),
    "classification_report": classification_report(labels, preds, output_dict=True),
    "confusion_matrix": confusion_matrix(labels, preds).tolist(),
}

print(classification_report(labels, preds))
print(confusion_matrix(labels, preds))

# Save test metrics
save_metrics(metrics)
