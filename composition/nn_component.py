import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class NNOutput:
    """Structured output from the neural network component."""
    raw_outputs: Dict[str, float]     # Named output values (e.g., {"risk_a": 0.82, "risk_b": 0.31})
    confidence: float                 # Overall prediction confidence
    model_name: str


class SimpleRiskNN(nn.Module):
    """A simple feedforward network for medical risk prediction.

    This is the toy NN for v0.1. It takes patient features and outputs
    risk scores for multiple conditions/treatments.

    Input: [age_normalized, blood_pressure_norm, glucose_norm, bmi_norm, ...]
    Output: [risk_treatment_a, risk_treatment_b, risk_interaction]
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output in [0, 1] range — interpretable as risk probabilities
        )
        self.output_names = [f"risk_{i}" for i in range(output_dim)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NNComponent:
    """Wraps a PyTorch model as a composition pipeline component.

    Usage:
        nn_comp = NNComponent(
            model=SimpleRiskNN(input_dim=8, output_dim=3),
            output_names=["risk_drug_a", "risk_drug_b", "risk_interaction"]
        )

        patient_data = torch.randn(1, 8)
        output = nn_comp.predict(patient_data)
        # output.raw_outputs = {"risk_drug_a": 0.82, "risk_drug_b": 0.31, "risk_interaction": 0.12}
    """

    def __init__(self, model: nn.Module, output_names: List[str] = None):
        self.model = model
        self.model.eval()
        self.output_names = output_names or [f"output_{i}" for i in range(100)]
        self.model_name = model.__class__.__name__

    def predict(self, input_tensor: torch.Tensor) -> NNOutput:
        """Run inference and return named outputs.

        Args:
            input_tensor: Shape (1, input_dim) or (input_dim,).

        Returns:
            NNOutput with named float values.
        """
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            raw = self.model(input_tensor).squeeze(0)

        outputs = {}
        for i, val in enumerate(raw.tolist()):
            name = self.output_names[i] if i < len(self.output_names) else f"output_{i}"
            outputs[name] = val

        confidence = float(torch.max(raw).item())

        return NNOutput(
            raw_outputs=outputs,
            confidence=confidence,
            model_name=self.model_name
        )

    def get_model(self) -> nn.Module:
        """Access underlying PyTorch model (for training)."""
        return self.model
