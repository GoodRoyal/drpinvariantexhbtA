import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional


class InvariantLoss(nn.Module):
    """Loss function that combines standard ML loss with structural invariant penalties.

    L_total = L_task + lambda * L_invariant

    Where L_invariant penalizes violations of detected structural invariants.
    This makes the AR layer guide ML training — the core of "AR-based ML."

    Usage:
        criterion = InvariantLoss(
            task_loss=nn.BCELoss(),
            invariant_weight=0.5
        )
        criterion.add_ordering_constraint("risk_drug_a", ">", "risk_drug_b", margin=0.1)
        criterion.add_bound_constraint("risk_interaction", 0.0, 0.5)

        loss = criterion(predictions, targets)
    """

    def __init__(self,
                 task_loss: nn.Module = None,
                 invariant_weight: float = 0.5):
        super().__init__()
        self.task_loss = task_loss or nn.BCELoss()
        self.invariant_weight = invariant_weight
        self.ordering_constraints: List[Dict] = []
        self.bound_constraints: List[Dict] = []

    def add_ordering_constraint(self, output_a: str, direction: str, output_b: str,
                                 idx_a: int = 0, idx_b: int = 1,
                                 margin: float = 0.05) -> None:
        """Add constraint that output_a should be > or < output_b.

        This encodes an ordering invariant from detection into a training constraint.

        Args:
            output_a/b: Names (for logging).
            direction: ">" or "<"
            idx_a/b: Indices into the model output tensor.
            margin: Minimum required difference (soft constraint).
        """
        self.ordering_constraints.append({
            "name_a": output_a, "name_b": output_b,
            "direction": direction,
            "idx_a": idx_a, "idx_b": idx_b,
            "margin": margin
        })

    def add_bound_constraint(self, output_name: str,
                              idx: int = 0,
                              lower: float = 0.0, upper: float = 1.0) -> None:
        """Add constraint that output should stay within bounds.

        Encodes a bounded oscillation invariant as a training constraint.
        """
        self.bound_constraints.append({
            "name": output_name, "idx": idx,
            "lower": lower, "upper": upper
        })

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predictions: Model output tensor, shape (batch, num_outputs).
            targets: Ground truth, shape (batch, num_outputs).

        Returns:
            Scalar loss tensor.
        """
        # Standard task loss
        loss_task = self.task_loss(predictions, targets)

        # Invariant violation penalties
        loss_invariant = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Ordering constraints
        for constraint in self.ordering_constraints:
            a = predictions[:, constraint["idx_a"]]
            b = predictions[:, constraint["idx_b"]]
            margin = constraint["margin"]

            if constraint["direction"] == ">":
                # Penalty when a is not sufficiently greater than b
                violation = torch.relu(b - a + margin)
            else:
                violation = torch.relu(a - b + margin)

            loss_invariant = loss_invariant + violation.mean()

        # Bound constraints
        for constraint in self.bound_constraints:
            val = predictions[:, constraint["idx"]]
            lower_violation = torch.relu(constraint["lower"] - val)
            upper_violation = torch.relu(val - constraint["upper"])
            loss_invariant = loss_invariant + (lower_violation + upper_violation).mean()

        return loss_task + self.invariant_weight * loss_invariant

    def get_violation_report(self, predictions: torch.Tensor) -> Dict[str, float]:
        """Check how many constraints are violated (for monitoring, not training).

        Returns dict mapping constraint_name -> fraction of batch violating.
        """
        report = {}
        with torch.no_grad():
            for c in self.ordering_constraints:
                a = predictions[:, c["idx_a"]]
                b = predictions[:, c["idx_b"]]
                if c["direction"] == ">":
                    violated = (a <= b).float().mean().item()
                else:
                    violated = (a >= b).float().mean().item()
                report[f"ordering_{c['name_a']}_{c['direction']}_{c['name_b']}"] = violated

            for c in self.bound_constraints:
                val = predictions[:, c["idx"]]
                violated = ((val < c["lower"]) | (val > c["upper"])).float().mean().item()
                report[f"bound_{c['name']}"] = violated

        return report
