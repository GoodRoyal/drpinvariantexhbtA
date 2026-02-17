import pytest
import torch
import torch.nn as nn
from training.invariant_loss import InvariantLoss


def make_preds(batch=8, outputs=3, val=0.5):
    return torch.full((batch, outputs), val, requires_grad=True)


def test_no_constraints_equals_task_loss():
    """With no invariant constraints, total loss should equal task loss."""
    criterion = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=1.0)
    preds = torch.rand(8, 3)
    targets = torch.rand(8, 3)
    total = criterion(preds, targets)
    task = nn.MSELoss()(preds, targets)
    assert torch.isclose(total, task)


def test_ordering_constraint_zero_when_satisfied():
    """Ordering penalty is zero when constraint is comfortably satisfied."""
    criterion = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=1.0)
    criterion.add_ordering_constraint("a", ">", "b", idx_a=0, idx_b=1, margin=0.05)

    # col 0 >> col 1, so a > b with plenty of margin
    preds = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.85, 0.15]])
    targets = torch.zeros_like(preds)
    loss_with = criterion(preds, targets)
    task_only = nn.MSELoss()(preds, targets)
    assert torch.isclose(loss_with, task_only)


def test_ordering_constraint_positive_when_violated():
    """Ordering penalty is positive when constraint is violated."""
    criterion = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=1.0)
    criterion.add_ordering_constraint("a", ">", "b", idx_a=0, idx_b=1, margin=0.05)

    # col 0 < col 1 — violated
    preds = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
    targets = torch.zeros_like(preds)
    loss_with = criterion(preds, targets)
    task_only = nn.MSELoss()(preds, targets)
    assert loss_with > task_only


def test_bound_constraint_zero_when_in_bounds():
    """Bound penalty is zero when all values are within bounds."""
    criterion = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=1.0)
    criterion.add_bound_constraint("x", idx=0, lower=0.0, upper=1.0)

    preds = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
    targets = torch.zeros_like(preds)
    loss_with = criterion(preds, targets)
    task_only = nn.MSELoss()(preds, targets)
    assert torch.isclose(loss_with, task_only)


def test_bound_constraint_positive_when_violated():
    """Bound penalty is positive when value exceeds upper bound."""
    criterion = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=1.0)
    criterion.add_bound_constraint("x", idx=0, lower=0.0, upper=0.3)

    preds = torch.tensor([[0.9, 0.5], [0.8, 0.5]])  # col 0 above upper=0.3
    targets = torch.zeros_like(preds)
    loss_with = criterion(preds, targets)
    task_only = nn.MSELoss()(preds, targets)
    assert loss_with > task_only


def test_invariant_weight_scales_penalty():
    """Higher invariant_weight should produce larger total loss on violations."""
    preds = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
    targets = torch.zeros_like(preds)

    c_low = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=0.1)
    c_low.add_ordering_constraint("a", ">", "b", idx_a=0, idx_b=1, margin=0.05)

    c_high = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=2.0)
    c_high.add_ordering_constraint("a", ">", "b", idx_a=0, idx_b=1, margin=0.05)

    assert c_high(preds, targets) > c_low(preds, targets)


def test_violation_report_keys():
    """get_violation_report should return keys for each constraint."""
    criterion = InvariantLoss()
    criterion.add_ordering_constraint("risk_a", ">", "risk_b", idx_a=0, idx_b=1)
    criterion.add_bound_constraint("risk_c", idx=2, lower=0.0, upper=0.5)

    preds = torch.rand(8, 3)
    report = criterion.get_violation_report(preds)

    assert "ordering_risk_a_>_risk_b" in report
    assert "bound_risk_c" in report


def test_violation_report_fraction_range():
    """Violation fractions must be in [0.0, 1.0]."""
    criterion = InvariantLoss()
    criterion.add_ordering_constraint("a", ">", "b", idx_a=0, idx_b=1)
    preds = torch.rand(16, 2)
    report = criterion.get_violation_report(preds)
    for v in report.values():
        assert 0.0 <= v <= 1.0


def test_loss_is_differentiable():
    """Loss must be differentiable so backward() works without error."""
    model = nn.Linear(4, 2)
    criterion = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=0.5)
    criterion.add_ordering_constraint("a", ">", "b", idx_a=0, idx_b=1, margin=0.1)

    x = torch.randn(8, 4)
    y = torch.rand(8, 2)
    loss = criterion(model(x), y)
    loss.backward()  # must not raise
