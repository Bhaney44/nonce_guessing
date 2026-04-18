"""
train.py
--------
Main entry point. Trains all three models on real Bitcoin block data
and evaluates their nonce prediction accuracy.

Usage:
    # First fetch data (skip if you already have data/blocks.json):
    python data_fetcher.py --blocks 10000

    # Train all models:
    python train.py

    # Train specific model only:
    python train.py --model mlp
    python train.py --model lstm
    python train.py --model transformer
"""

import argparse
import json
import numpy as np
from pathlib import Path

from features import load_blocks, engineer_features, prepare_datasets, make_sequence_dataset
from models import build_model
from trainer import Trainer
from evaluate import compute_metrics, print_metrics, plot_all_results


SEQ_LEN = 50       # sequence window for LSTM/Transformer
EPOCHS  = 150
BATCH   = 256


def train_mlp(feat, args):
    print("\n" + "="*60)
    print("  MODEL: MLP")
    print("="*60)

    (
        X_train, y_train_s, y_train,
        X_val,   y_val_s,   y_val,
        X_test,  y_test_s,  y_test,
        feature_cols, y_mean, y_std,
    ) = prepare_datasets(feat)

    model = build_model("mlp", input_dim=X_train.shape[1])
    trainer = Trainer(model, model_name="mlp")
    history = trainer.fit(
        X_train, y_train_s, X_val, y_val_s,
        epochs=EPOCHS, batch_size=BATCH,
    )

    # Predictions in original scale
    y_pred_s = trainer.predict(X_test)
    y_pred = y_pred_s * y_std + y_mean

    metrics = compute_metrics(y_test, y_pred, y_std)
    print_metrics(metrics, "MLP")
    trainer.save_history("results/mlp_history.json")

    return {"metrics": metrics, "history": history, "y_true": y_test, "y_pred": y_pred}


def train_lstm(feat, args):
    print("\n" + "="*60)
    print("  MODEL: LSTM")
    print("="*60)

    X_seq, y_seq, feature_cols, scaler = make_sequence_dataset(feat, seq_len=SEQ_LEN)

    n = len(X_seq)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    n_train = n - n_test - n_val

    X_train, X_val, X_test = X_seq[:n_train], X_seq[n_train:n_train+n_val], X_seq[n_train+n_val:]
    y_train, y_val, y_test = y_seq[:n_train], y_seq[n_train:n_train+n_val], y_seq[n_train+n_val:]

    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std
    y_val_s   = (y_val - y_mean) / y_std

    model = build_model("lstm", input_dim=X_seq.shape[2], seq_len=SEQ_LEN)
    trainer = Trainer(model, model_name="lstm")
    history = trainer.fit(
        X_train, y_train_s, X_val, y_val_s,
        epochs=EPOCHS, batch_size=BATCH, is_sequence=True,
    )

    y_pred_s = trainer.predict(X_test)
    y_pred = y_pred_s * y_std + y_mean

    metrics = compute_metrics(y_test, y_pred, y_std)
    print_metrics(metrics, "LSTM")
    trainer.save_history("results/lstm_history.json")

    return {"metrics": metrics, "history": history, "y_true": y_test, "y_pred": y_pred}


def train_transformer(feat, args):
    print("\n" + "="*60)
    print("  MODEL: Transformer")
    print("="*60)

    X_seq, y_seq, feature_cols, scaler = make_sequence_dataset(feat, seq_len=SEQ_LEN)

    n = len(X_seq)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    n_train = n - n_test - n_val

    X_train, X_val, X_test = X_seq[:n_train], X_seq[n_train:n_train+n_val], X_seq[n_train+n_val:]
    y_train, y_val, y_test = y_seq[:n_train], y_seq[n_train:n_train+n_val], y_seq[n_train+n_val:]

    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std
    y_val_s   = (y_val - y_mean) / y_std

    model = build_model("transformer", input_dim=X_seq.shape[2], seq_len=SEQ_LEN)
    trainer = Trainer(model, model_name="transformer")
    history = trainer.fit(
        X_train, y_train_s, X_val, y_val_s,
        epochs=EPOCHS, batch_size=BATCH, is_sequence=True,
        lr=3e-4,  # Transformers prefer lower LR
    )

    y_pred_s = trainer.predict(X_test)
    y_pred = y_pred_s * y_std + y_mean

    metrics = compute_metrics(y_test, y_pred, y_std)
    print_metrics(metrics, "Transformer")
    trainer.save_history("results/transformer_history.json")

    return {"metrics": metrics, "history": history, "y_true": y_test, "y_pred": y_pred}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "mlp", "lstm", "transformer"],
                        help="Which model(s) to train")
    parser.add_argument("--data", type=str, default="data/blocks.json",
                        help="Path to blocks data JSON")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    # ── Load & engineer features ──────────────────────────────────────────
    print(f"Loading data from {args.data}...")
    df = load_blocks(args.data)
    n_blocks = len(df)
    print(f"  {n_blocks:,} blocks loaded (heights {df['height'].min()}–{df['height'].max()})")

    MIN_BLOCKS = 500
    if n_blocks < MIN_BLOCKS:
        print(f"\n  ✗ Not enough data: need at least {MIN_BLOCKS} blocks, have {n_blocks}.")
        print(f"  Run: python data_fetcher.py --blocks 5000")
        print(f"  The fetcher is resumable — it will pick up where it left off.")
        return

    feat = engineer_features(df)
    print(f"  Feature matrix: {feat.shape[0]:,} samples × {feat.shape[1]} features")

    # ── Train ─────────────────────────────────────────────────────────────
    all_results = {}

    if args.model in ("all", "mlp"):
        all_results["MLP"] = train_mlp(feat, args)

    if args.model in ("all", "lstm"):
        all_results["LSTM"] = train_lstm(feat, args)

    if args.model in ("all", "transformer"):
        all_results["Transformer"] = train_transformer(feat, args)

    # ── Plot ─────────────────────────────────────────────────────────────
    if all_results:
        plot_all_results(all_results, output_path="results/analysis.png")

    # Save summary
    summary = {
        name: {
            "metrics": {
                k: v for k, v in res["metrics"].items()
                if k not in ("within_range",)
            },
            "within_range": res["metrics"]["within_range"],
        }
        for name, res in all_results.items()
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to results/summary.json")


if __name__ == "__main__":
    main()
