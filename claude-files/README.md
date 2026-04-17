# Bitcoin Nonce Predictor — Neural Network Research

Predicts Bitcoin mining nonces from real historical block data.

Inspired by: https://github.com/Bhaney44/nonce_guessing

## Core Idea

Rather than trying to reverse-engineer SHA-256 (cryptographically unsound),
this project asks: **do winning nonces from real mined blocks have statistical
patterns we can learn?**

This is a legitimate question because:
- Miners search nonces in different ways (sequential, random, hardware-specific)
- Mining pools may have characteristic search strategies
- Difficulty adjustments correlate with which nonces miners reached
- Historical nonce distributions have observable structure (not uniform)

## Files

| File | Purpose |
|---|---|
| `data_fetcher.py` | Pulls real block headers from blockchain.info |
| `features.py` | Engineers lag/rolling/time/difficulty features |
| `models.py` | MLP, LSTM with attention, Transformer |
| `trainer.py` | Training loop: early stopping, cosine annealing, checkpointing |
| `evaluate.py` | Metrics: MAE, improvement vs random, narrowing factor, plots |
| `train.py` | Main runner |

## Quickstart

```bash
pip install -r requirements.txt

# Step 1: Fetch real Bitcoin block data
python data_fetcher.py --blocks 10000

# Step 2: Train all models
python train.py

# Or train one at a time:
python train.py --model mlp
python train.py --model lstm
python train.py --model transformer
```

## Key Metrics

**Improvement vs random** — how much better is MAE than a random guess (random
guess has expected error of 2³² / 4 ≈ 1 billion)? A value of 2.0x means the
model's average error is half of random.

**Narrowing factor** — if you search outward from the model's prediction, how
much of the 2³² nonce space do you skip? Factor = 2³² / (2 × median_error).
Any factor > 1 means the model is saving hashes.

**Within-range accuracy** — what fraction of predictions land within 1%, 5%,
10%, 25%, 50% of the true nonce?

## Feature Engineering

- **Lag features**: nonce[t-1] through nonce[t-21] (Fibonacci spacing)
- **Rolling stats**: mean, std, min, max, range over windows of 3/7/21/50/100 blocks
- **Difficulty**: decoded from `bits` field, log-transformed, change rate
- **Time**: inter-block delta (log), hour-of-day and day-of-week (sin/cos encoded)
- **Header fields**: version, n_tx, block size

## Architecture

**MLP**: 4-layer feedforward [512→256→128→64→1] with LayerNorm + GELU + Dropout

**LSTM**: 3-layer bidirectional LSTM (256 hidden) + multi-head self-attention + pooling

**Transformer**: 4-layer encoder with positional encoding, Pre-LN, GELU, 8 attention heads

All models trained with AdamW + Huber loss + cosine annealing + early stopping.

## Data

Data comes from [blockchain.info](https://blockchain.info) API, which provides
publicly available Bitcoin block header data. No API key required. The fetcher
is rate-limited to be polite to the server.

~10,000 blocks ≈ 70 days of Bitcoin history, providing ample training data.
For serious experiments, use 100,000+ blocks (all blocks since genesis).
