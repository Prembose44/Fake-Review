"""
Step 4: Fine-tune DeBERTa-v3-base (Tier 2 of the cascade, and the pure-transformer baseline).

Trained on Maxwell-only data (clean, validated CG/OR labels).
Sized for a 6GB VRAM GPU (e.g. RTX 4050) using bf16 + gradient accumulation.

Key fixes baked in from prior debugging:
  - dtype=torch.float32 at load time (fp16 caused gradient underflow)
  - bf16=True, fp16=False (fp16 + DeBERTa-v3 has a known gradient-scaler bug)
  - learning_rate=1e-5 (DeBERTa-v3 is sensitive; 2e-5 caused instability)
  - max_grad_norm=1.0 (explicit gradient clipping)

Usage (from project root, with venv activated):
    python src\\04_train_deberta.py
"""

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATA_DIR = "data/processed"
MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = "models/deberta_v3_base"
MAX_LENGTH = 256
RANDOM_SEED = 42


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).dropna(subset=["text"])
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv")).dropna(subset=["text"])
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).dropna(subset=["text"])
    return train, val, test


def tokenize_dataset(df, tokenizer):
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length",
        )

    ds = ds.map(tokenize_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected, this will be very slow on CPU.")

    print("\nLoading data...")
    train_df, val_df, test_df = load_data()
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, dtype=torch.float32,
    )
    print(f"Model dtype after load: {next(model.parameters()).dtype}")

    print("\nTokenising datasets (this may take a minute)...")
    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds = tokenize_dataset(val_df, tokenizer)
    test_ds = tokenize_dataset(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    trainer.train()

    print("\n--- Final validation metrics ---")
    val_metrics = trainer.evaluate(val_ds)
    print(val_metrics)

    print("\n--- Test set metrics ---")
    test_metrics = trainer.evaluate(test_ds)
    print(test_metrics)

    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved final model + tokenizer to {final_path}/")

    print("\nDone. This model is your Tier 2 classifier and pure-transformer baseline.")


if __name__ == "__main__":
    main()
