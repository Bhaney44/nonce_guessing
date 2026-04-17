"""
trainer.py
----------
Training loop with early stopping, cosine annealing, gradient clipping,
and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import json
import time


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "model",
        device: str = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {"train_loss": [], "val_loss": [], "epochs": []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        min_delta: float = 1e-5,
        is_sequence: bool = False,
    ) -> dict:
        """Train with early stopping and cosine annealing."""

        # Build DataLoaders
        if is_sequence:
            # X shape: (batch, seq_len, features)
            X_tr = torch.tensor(X_train, dtype=torch.float32)
            X_v  = torch.tensor(X_val,   dtype=torch.float32)
        else:
            X_tr = torch.tensor(X_train, dtype=torch.float32)
            X_v  = torch.tensor(X_val,   dtype=torch.float32)

        y_tr = torch.tensor(y_train, dtype=torch.float32)
        y_v  = torch.tensor(y_val,   dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_v, y_v), batch_size=batch_size * 4
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.HuberLoss(delta=1.0)

        best_val = float("inf")
        patience_counter = 0
        best_epoch = 0

        print(f"\n{'─'*60}")
        print(f"  Training {self.model_name} on {self.device}")
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Params: {self._count_params():,}")
        print(f"{'─'*60}")

        t_start = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                pred = self.model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_b)

            train_loss /= len(X_train)

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    pred = self.model(X_b)
                    val_loss += criterion(pred, y_b).item() * len(X_b)
            val_loss /= len(X_val)

            scheduler.step()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["epochs"].append(epoch)

            # Early stopping + checkpointing
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1 or patience_counter == patience:
                elapsed = time.time() - t_start
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch:4d}/{epochs} | "
                    f"train={train_loss:.5f} | val={val_loss:.5f} | "
                    f"best={best_val:.5f} (ep{best_epoch}) | "
                    f"lr={lr_now:.2e} | {elapsed:.0f}s"
                )

            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (best: epoch {best_epoch})")
                break

        # Load best checkpoint
        self._load_best_checkpoint()
        return self.history

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=batch_size)
        preds = []
        with torch.no_grad():
            for (X_b,) in loader:
                X_b = X_b.to(self.device)
                preds.append(self.model(X_b).cpu().numpy())
        return np.concatenate(preds)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _save_checkpoint(self, epoch: int, val_loss: float):
        path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": self.model.state_dict(),
        }, path)

    def _load_best_checkpoint(self):
        path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])

    def save_history(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
