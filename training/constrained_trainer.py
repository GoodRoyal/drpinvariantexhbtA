import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Callable
from training.invariant_loss import InvariantLoss


class ConstrainedTrainer:
    """Trains a model while enforcing structural invariant constraints.

    This is AR-based ML training: the AR scaffold (invariant constraints derived
    from the logic program) shapes how the NN learns.

    Usage:
        trainer = ConstrainedTrainer(
            model=my_nn,
            criterion=my_invariant_loss,
            lr=0.001,
            epochs=50
        )
        history = trainer.train(train_loader)
        # history contains loss curves and invariant violation rates per epoch
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: InvariantLoss,
                 lr: float = 0.001,
                 epochs: int = 50,
                 device: str = "cpu"):
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.epochs = epochs
        self.device = device
        self.model.to(device)

    def train(self, train_loader: DataLoader,
              val_loader: DataLoader = None,
              verbose: bool = True) -> Dict[str, List]:
        """Train the model with invariant constraints.

        Args:
            train_loader: Training data (features, labels).
            val_loader: Optional validation data.
            verbose: Print progress per epoch.

        Returns:
            History dict with keys:
                "train_loss", "val_loss", "invariant_violations"
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "invariant_violations": [],
        }

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_violations = {}
            num_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Track violations
                violations = self.criterion.get_violation_report(preds)
                for k, v in violations.items():
                    epoch_violations[k] = epoch_violations.get(k, 0) + v
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_violations = {k: v / num_batches for k, v in epoch_violations.items()}

            history["train_loss"].append(avg_loss)
            history["invariant_violations"].append(avg_violations)

            # Validation
            if val_loader:
                val_loss = self._evaluate(val_loader)
                history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                viol_str = ", ".join(f"{k}: {v:.1%}" for k, v in avg_violations.items())
                print(f"Epoch {epoch+1}/{self.epochs} — loss: {avg_loss:.4f} — violations: {viol_str}")

        return history

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                total_loss += loss.item()
                count += 1
        return total_loss / max(count, 1)

    @staticmethod
    def make_loader(features: torch.Tensor, labels: torch.Tensor,
                    batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Helper to create DataLoader from tensors."""
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
