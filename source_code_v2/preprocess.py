"""Text preprocessing for Reddit and Twitter sarcasm detection."""

# Reddit: reply, parent, label. Twitter: tweet, sarcastic.
# Reddit with use_context=True: "[CTX] parent [REPLY] reply".

import pandas as pd
import emoji
import re


def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")


def clean_text(text):
    text = str(text)

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    return text.strip()


DATA_PATHS = {
    "reddit": {
        "train": "../reddit_data/train.csv",
        "val": "../reddit_data/val.csv",
        "test": "../reddit_data/test.csv",
    },
    "twitter": {
        "train": "../twitter_data/train.csv",
        "val": "../twitter_data/val.csv",
        "test": "../twitter_data/test.csv",
    },
}


def make_experiment_name(dataset, use_context=False, remove_emoji=False):
    context_part = "context" if use_context else "no_context"
    emoji_part = "no_emoji" if remove_emoji else "emoji"
    return f"{dataset}_{context_part}_{emoji_part}"


def default_max_length(use_context=False):
    return 256 if use_context else 128


def _has_parent(parent):
    if parent is None or (isinstance(parent, float) and pd.isna(parent)):
        return False
    return bool(str(parent).strip())


def _maybe_remove_emojis(text, remove_emoji):
    if remove_emoji:
        return remove_emojis(text)
    return text


def _assign_labels(df):
    if "sarcastic" in df.columns:
        df["labels"] = df["sarcastic"].astype(int)
    elif "label" in df.columns:
        df["labels"] = df["label"].astype(int)


def build_input(reply, parent=None, use_context=False):
    if use_context and _has_parent(parent):
        return f"[CTX] {parent} [REPLY] {reply}"

    return reply


def preprocess_dataframe(df, use_context=False, remove_emoji=False):
    """Build text and labels columns."""
    df = df.copy()
    columns = df.columns

    is_reddit = "reply" in columns and "parent" in columns
    is_twitter = "tweet" in columns and "sarcastic" in columns

    if not is_reddit and not is_twitter:
        raise ValueError(f"Unknown dataframe columns: {columns}")

    processed_texts = []

    for _, row in df.iterrows():
        if is_reddit:
            reply = clean_text(row["reply"])
            parent = clean_text(row["parent"]) if _has_parent(row["parent"]) else None

            reply = _maybe_remove_emojis(reply, remove_emoji)
            if parent is not None:
                parent = _maybe_remove_emojis(parent, remove_emoji)

            text = build_input(reply, parent, use_context=use_context)
        else:
            text = clean_text(row["tweet"])
            text = _maybe_remove_emojis(text, remove_emoji)

        processed_texts.append(text)

    df["text"] = processed_texts
    _assign_labels(df)

    return df
