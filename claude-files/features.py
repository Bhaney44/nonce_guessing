"""
features.py
-----------
Feature engineering from real Bitcoin block header data.

Key insight from the reference repo:
  - Winning nonces from real blocks may have statistical structure
    (miner strategies, hardware quirks, search patterns)
  - Other header fields (version, bits, time delta, merkle diversity)
    may correlate with the search space that miners actually explored
  - Moving averages of nonces (like the repo's avg3/avg21) are baseline features

Feature set:
  - Lag features: nonce[t-1], nonce[t-2], ..., nonce[t-k]
  - Rolling statistics: mean, std, min, max over windows
  - Header fields: version, bits (difficulty), time_delta
  - Derived: nonce range exhaustion proxy (nonce / 2^32)
  - Cyclical time encoding: hour-of-day, day-of-week sin/cos
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


LAG_STEPS = [1, 2, 3, 5, 8, 13, 21]   # Fibonacci lags
ROLL_WINDOWS = [3, 7, 21, 50, 100]


def load_blocks(path: str = "data/blocks.json") -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["blocks"])

    # Sort chronologically (blocks are stored newest-first from fetcher)
    df = df.sort_values("height").reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from raw block data."""
    feat = pd.DataFrame(index=df.index)

    # ── Core fields ──────────────────────────────────────────────────────────
    feat["nonce"] = df["nonce"].astype(np.float64)
    feat["version"] = df["version"].astype(np.float64)
    feat["bits"] = df["bits"].astype(np.float64)
    feat["n_tx"] = df["n_tx"].astype(np.float64)
    feat["size"] = df["size"].astype(np.float64)

    # Nonce normalized to [0, 1] — how far through the 32-bit space did miner search?
    feat["nonce_normalized"] = feat["nonce"] / (2**32 - 1)

    # ── Time features ─────────────────────────────────────────────────────────
    feat["timestamp"] = df["time"].astype(np.float64)
    feat["time_delta"] = feat["timestamp"].diff().fillna(600)  # ~600s average block time
    feat["time_delta_log"] = np.log1p(feat["time_delta"].clip(lower=1))

    # Cyclical time encoding
    hours = pd.to_datetime(df["time"], unit="s").dt.hour
    feat["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    dow = pd.to_datetime(df["time"], unit="s").dt.dayofweek
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in LAG_STEPS:
        feat[f"nonce_lag_{lag}"] = feat["nonce"].shift(lag)
        feat[f"nonce_norm_lag_{lag}"] = feat["nonce_normalized"].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in ROLL_WINDOWS:
        roll = feat["nonce"].shift(1).rolling(window=w)
        feat[f"roll_mean_{w}"] = roll.mean()
        feat[f"roll_std_{w}"] = roll.std()
        feat[f"roll_min_{w}"] = roll.min()
        feat[f"roll_max_{w}"] = roll.max()
        feat[f"roll_range_{w}"] = feat[f"roll_max_{w}"] - feat[f"roll_min_{w}"]

    # ── Difficulty-derived features ───────────────────────────────────────────
    # bits encodes the target compactly — decode to approximate difficulty
    def bits_to_difficulty(b):
        exp = (b >> 24) & 0xFF
        mant = b & 0x007FFFFF
        target = mant * (2 ** (8 * (exp - 3)))
        genesis_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
        return genesis_target / max(target, 1)

    feat["difficulty"] = df["bits"].apply(bits_to_difficulty)
    feat["difficulty_log"] = np.log1p(feat["difficulty"])
    feat["difficulty_change"] = feat["difficulty_log"].diff().fillna(0)

    # ── Target ───────────────────────────────────────────────────────────────
    feat["target_nonce"] = feat["nonce"]  # predict current nonce from history

    # Drop rows with NaN (from lag/rolling)
    max_lag = max(LAG_STEPS + ROLL_WINDOWS)
    feat = feat.iloc[max_lag:].reset_index(drop=True)

    return feat


def prepare_datasets(
    feat: pd.DataFrame,
    target_col: str = "target_nonce",
    test_size: float = 0.15,
    val_size: float = 0.15,
    scaler_path: str = "data/scaler.pkl",
) -> tuple:
    """
    Split into train/val/test. Uses time-ordered split (no shuffling —
    we want to predict future blocks from past blocks).
    """
    # Drop non-feature columns
    drop_cols = [target_col, "timestamp", "nonce"]
    feature_cols = [c for c in feat.columns if c not in drop_cols]

    X = feat[feature_cols].values.astype(np.float32)
    y = feat[target_col].values.astype(np.float32)

    n = len(X)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]

    # Scale X (fit only on training data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Scale y: nonce is 0–2^32, normalize to roughly N(0,1)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std
    y_val_s   = (y_val - y_mean) / y_std
    y_test_s  = (y_test - y_mean) / y_std

    # Save scaler and y stats
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "y_mean": y_mean, "y_std": y_std}, scaler_path)

    return (
        X_train, y_train_s, y_train,
        X_val,   y_val_s,   y_val,
        X_test,  y_test_s,  y_test,
        feature_cols,
        y_mean, y_std,
    )


def make_sequence_dataset(
    feat: pd.DataFrame,
    seq_len: int = 50,
    target_col: str = "target_nonce",
) -> tuple:
    """
    Build (batch, seq_len, n_features) tensors for LSTM/Transformer.
    Each sample is a window of seq_len blocks, predicting the next nonce.
    """
    drop_cols = [target_col, "timestamp", "nonce"]
    feature_cols = [c for c in feat.columns if c not in drop_cols]

    X_raw = feat[feature_cols].values.astype(np.float32)
    y_raw = feat[target_col].values.astype(np.float32)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i - seq_len:i])
        y_seq.append(y_raw[i])

    return np.array(X_seq), np.array(y_seq), feature_cols, scaler
