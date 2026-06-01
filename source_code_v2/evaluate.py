import os
import json
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from preprocess import preprocess_dataframe


EXPERIMENT_NAME = "twitter_no_context_emoji"

MODEL_PATH = f"./best_model_{EXPERIMENT_NAME}"
TEST_PATH = "../twitter_data/test.csv"

USE_CONTEXT = False
REMOVE_EMOJI = False
MAX_LENGTH = 128


def build_dataset(df, tokenizer):
    """Preprocess and tokenize the test split."""
    df = preprocess_dataframe(
        df,
        use_context=USE_CONTEXT,
        remove_emoji=REMOVE_EMOJI
    )

    dataset = Dataset.from_pandas(df[["text", "labels"]])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return df, dataset


def save_metrics(metrics):
    """Save test metrics inside the model folder."""
    output_path = os.path.join(MODEL_PATH, "test_metrics.json")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)


test_df = pd.read_csv(TEST_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

test_df, test_dataset = build_dataset(test_df, tokenizer)

trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset)

preds = torch.argmax(torch.tensor(predictions.predictions), dim=-1).numpy()
labels = test_df["labels"].values

metrics = {
    "experiment": EXPERIMENT_NAME,
    "test_f1": f1_score(labels, preds),
    "classification_report": classification_report(labels, preds, output_dict=True),
    "confusion_matrix": confusion_matrix(labels, preds).tolist()
}

print(classification_report(labels, preds))
print(confusion_matrix(labels, preds))

save_metrics(metrics)