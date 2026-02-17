import pytest
import torch
import torch.nn as nn
from training.invariant_loss import InvariantLoss
from training.constrained_trainer import ConstrainedTrainer
from composition.nn_component import SimpleRiskNN


def make_trainer(epochs=3, invariant_weight=0.5):
    model = SimpleRiskNN(input_dim=4, output_dim=2)
    criterion = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=invariant_weight)
    criterion.add_ordering_constraint("risk_a", ">", "risk_b", idx_a=0, idx_b=1, margin=0.05)
    return ConstrainedTrainer(model=model, criterion=criterion, lr=0.01, epochs=epochs)


def make_loader(n=64, input_dim=4, output_dim=2, batch_size=16):
    X = torch.rand(n, input_dim)
    y = torch.rand(n, output_dim)
    return ConstrainedTrainer.make_loader(X, y, batch_size=batch_size)


def test_train_returns_history_keys():
    """train() history should contain the three standard keys."""
    trainer = make_trainer(epochs=2)
    loader = make_loader()
    history = trainer.train(loader, verbose=False)

    assert "train_loss" in history
    assert "val_loss" in history
    assert "invariant_violations" in history


def test_train_loss_length_matches_epochs():
    """train_loss list should have one entry per epoch."""
    epochs = 4
    trainer = make_trainer(epochs=epochs)
    history = trainer.train(make_loader(), verbose=False)
    assert len(history["train_loss"]) == epochs


def test_val_loss_populated_when_loader_given():
    """val_loss should be populated when val_loader is provided."""
    trainer = make_trainer(epochs=3)
    history = trainer.train(make_loader(), val_loader=make_loader(n=32), verbose=False)
    assert len(history["val_loss"]) == 3


def test_val_loss_empty_without_loader():
    """val_loss should be empty list when no val_loader is given."""
    trainer = make_trainer(epochs=2)
    history = trainer.train(make_loader(), verbose=False)
    assert history["val_loss"] == []


def test_invariant_violations_tracked_per_epoch():
    """invariant_violations should be a list of dicts, one per epoch."""
    trainer = make_trainer(epochs=3)
    history = trainer.train(make_loader(), verbose=False)
    assert len(history["invariant_violations"]) == 3
    for entry in history["invariant_violations"]:
        assert isinstance(entry, dict)


def test_train_loss_is_positive():
    """All recorded train losses should be positive."""
    trainer = make_trainer(epochs=3)
    history = trainer.train(make_loader(), verbose=False)
    for loss in history["train_loss"]:
        assert loss > 0.0


def test_make_loader_produces_dataloader():
    """make_loader static helper should return a usable DataLoader."""
    from torch.utils.data import DataLoader
    X = torch.rand(20, 4)
    y = torch.rand(20, 2)
    loader = ConstrainedTrainer.make_loader(X, y, batch_size=5)
    assert isinstance(loader, DataLoader)
    batches = list(loader)
    assert len(batches) == 4  # 20 / 5


def test_model_parameters_update_during_training():
    """Model weights should change after training (gradient flow confirmed)."""
    model = SimpleRiskNN(input_dim=4, output_dim=2)
    params_before = [p.clone() for p in model.parameters()]

    criterion = InvariantLoss(task_loss=nn.MSELoss(), invariant_weight=0.5)
    trainer = ConstrainedTrainer(model=model, criterion=criterion, lr=0.1, epochs=3)
    trainer.train(make_loader(), verbose=False)

    params_after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    assert changed
