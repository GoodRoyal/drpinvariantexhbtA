"""Smoke test for training experiment — just verify it runs without errors."""
import pytest


def test_training_experiment_runs():
    """The experiment should complete without errors on small sizes."""
    from examples.training_experiment import generate_medical_data, evaluate_invariant_compliance, run_experiment

    # Just test with tiny sizes to verify wiring
    results = run_experiment(sample_sizes=[20, 50], num_trials=1)

    assert len(results["constrained"]) == 2
    assert len(results["unconstrained"]) == 2

    # Constrained should have ordering compliance >= 50% even at N=20
    assert results["constrained"][0][0] >= 0.5
