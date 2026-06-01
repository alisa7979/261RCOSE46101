"""Train RoBERTa for sarcasm detection. See Experiment config below."""

import os
import json

import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from preprocess import (
    DATA_PATHS,
    default_max_length,
    make_experiment_name,
    preprocess_dataframe,
)


# Experiment config
# Change these constants to switch runs.
DATASET = "reddit"  # "reddit" or "twitter"
USE_CONTEXT = True  # Reddit only; ignored for Twitter
REMOVE_EMOJI = False  # True = strip emojis, False = keep emojis

# Optional override; leave None to auto-generate from flags above
EXPERIMENT_NAME = None

# Model and training settings
MODEL_NAME = "roberta-base"
CLASS_WEIGHTS = (1.0, 3.0)

if DATASET == "twitter" and USE_CONTEXT:
    USE_CONTEXT = False

EXPERIMENT_NAME = EXPERIMENT_NAME or make_experiment_name(
    DATASET, use_context=USE_CONTEXT, remove_emoji=REMOVE_EMOJI
)

# Derived paths and experiment name
TRAIN_PATH = DATA_PATHS[DATASET]["train"]
VAL_PATH = DATA_PATHS[DATASET]["val"]
MAX_LENGTH = default_max_length(USE_CONTEXT)

OUTPUT_DIR = f"./results_{EXPERIMENT_NAME}"
BEST_MODEL_DIR = f"./best_model_{EXPERIMENT_NAME}"


# Load data
train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8")
val_df = pd.read_csv(VAL_PATH, encoding="utf-8")

# Preprocess
train_df = preprocess_dataframe(
    train_df,
    use_context=USE_CONTEXT,
    remove_emoji=REMOVE_EMOJI
)

val_df = preprocess_dataframe(
    val_df,
    use_context=USE_CONTEXT,
    remove_emoji=REMOVE_EMOJI
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Tokenize and build datasets
def tokenize(batch):
    """Tokenize text inputs for RoBERTa."""
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


train_dataset = Dataset.from_pandas(train_df[["text", "labels"]])
val_dataset = Dataset.from_pandas(val_df[["text", "labels"]])

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


def compute_metrics(eval_pred):
    """Compute evaluation metrics after each validation run."""
    logits, labels = eval_pred

    predictions = torch.argmax(
        torch.tensor(logits),
        dim=-1
    )

    print("Pred counts:", torch.bincount(predictions))

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }


class WeightedTrainer(Trainer):
    """Custom trainer with weighted cross-entropy for class imbalance."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs["logits"]

        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(CLASS_WEIGHTS).to(logits.device)
        )

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Training setup
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_only_model=True,
    save_total_limit=1,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train and save checkpoint
train_result = trainer.train()

trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

metrics = trainer.evaluate()
metrics["experiment"] = EXPERIMENT_NAME
metrics["dataset"] = DATASET
metrics["model_name"] = MODEL_NAME
metrics["use_context"] = USE_CONTEXT
metrics["remove_emoji"] = REMOVE_EMOJI
metrics["max_length"] = MAX_LENGTH
metrics["class_weights"] = CLASS_WEIGHTS
metrics["train_runtime"] = train_result.metrics.get("train_runtime")
metrics["train_loss"] = train_result.metrics.get("train_loss")

with open(os.path.join(BEST_MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)