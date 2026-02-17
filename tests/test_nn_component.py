import pytest
import torch
from composition.nn_component import NNComponent, NNOutput, SimpleRiskNN


def test_predict_returns_named_outputs():
    """NNComponent.predict should return a dict keyed by output_names."""
    model = SimpleRiskNN(input_dim=4, output_dim=3)
    comp = NNComponent(model, output_names=["risk_a", "risk_b", "risk_c"])
    x = torch.zeros(4)
    result = comp.predict(x)

    assert isinstance(result, NNOutput)
    assert set(result.raw_outputs.keys()) == {"risk_a", "risk_b", "risk_c"}


def test_predict_values_in_unit_interval():
    """SimpleRiskNN uses Sigmoid, so all outputs must be in [0, 1]."""
    model = SimpleRiskNN(input_dim=8, output_dim=3)
    comp = NNComponent(model)
    x = torch.randn(8)
    result = comp.predict(x)

    for name, val in result.raw_outputs.items():
        assert 0.0 <= val <= 1.0, f"{name}={val} not in [0,1]"


def test_predict_accepts_1d_and_2d_input():
    """predict() should handle both (input_dim,) and (1, input_dim) shapes."""
    model = SimpleRiskNN(input_dim=4, output_dim=2)
    comp = NNComponent(model, output_names=["a", "b"])

    out1 = comp.predict(torch.zeros(4))
    out2 = comp.predict(torch.zeros(1, 4))

    assert out1.raw_outputs == out2.raw_outputs


def test_confidence_equals_max_output():
    """confidence should be the maximum raw output value."""
    model = SimpleRiskNN(input_dim=4, output_dim=3)
    comp = NNComponent(model, output_names=["x", "y", "z"])
    result = comp.predict(torch.zeros(4))

    assert result.confidence == max(result.raw_outputs.values())


def test_model_name_reflects_class():
    """model_name should be the class name of the wrapped model."""
    model = SimpleRiskNN()
    comp = NNComponent(model)
    assert comp.model_name == "SimpleRiskNN"


def test_get_model_returns_module():
    """get_model() should return the underlying nn.Module."""
    model = SimpleRiskNN()
    comp = NNComponent(model)
    assert comp.get_model() is model


def test_default_output_names_when_none():
    """When output_names is None, names should fall back to 'output_N'."""
    model = SimpleRiskNN(input_dim=4, output_dim=2)
    comp = NNComponent(model)
    result = comp.predict(torch.zeros(4))

    assert "output_0" in result.raw_outputs
    assert "output_1" in result.raw_outputs
