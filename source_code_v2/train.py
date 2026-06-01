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

from preprocess import preprocess_dataframe


# Model + preprocessing config
MODEL_NAME = "roberta-base"
USE_CONTEXT = False
REMOVE_EMOJI = False

# Data paths
TRAIN_PATH = "../twitter_data/train.csv"
VAL_PATH = "../twitter_data/val.csv"

# Experiment config
EXPERIMENT_NAME = "twitter_no_context_emoji"

# Output config
OUTPUT_DIR = f"./results_{EXPERIMENT_NAME}"
BEST_MODEL_DIR = f"./best_model_{EXPERIMENT_NAME}"

# Training constants
MAX_LENGTH = 128
CLASS_WEIGHTS = (1.0, 3.0)


train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

# Preprocess datasets
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
    save_total_limit=1,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
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

train_result = trainer.train()

trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

metrics = trainer.evaluate()
metrics["experiment"] = EXPERIMENT_NAME
metrics["model_name"] = MODEL_NAME
metrics["use_context"] = USE_CONTEXT
metrics["remove_emoji"] = REMOVE_EMOJI
metrics["max_length"] = MAX_LENGTH
metrics["class_weights"] = CLASS_WEIGHTS
metrics["train_runtime"] = train_result.metrics.get("train_runtime")
metrics["train_loss"] = train_result.metrics.get("train_loss")

with open(os.path.join(BEST_MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)