import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.invariant_detector import InvariantDetector, Invariant
from core.lossy_translation import LossyTranslator, TranslationChain, TranslationRecord
from core.coordination_controller import CoordinationController, CoordinationEvent
from core.proof_generator import ProofGenerator, ProofNode
from composition.nn_component import NNComponent, NNOutput
from composition.problog_component import ProbLogComponent, LPOutput


@dataclass
class PipelineResult:
    """Complete result from one pass through the composition pipeline."""
    nn_output: NNOutput
    translation_records: List[TranslationRecord]
    translated_values: Dict[str, float]        # After lossy translation
    lp_output: LPOutput
    coordination_events: List[CoordinationEvent]

    def summary(self) -> str:
        lines = [
            f"=== Pipeline Result ===",
            f"NN outputs: {self.nn_output.raw_outputs}",
            f"After translation: {self.translated_values}",
            f"LP results: {self.lp_output.query_results}",
            f"Coordination events: {len(self.coordination_events)}",
        ]
        for event in self.coordination_events:
            lines.append(f"  [{event.action.value}] {event.message}")
        return "\n".join(lines)


class CompositionPipeline:
    """Full NN → Lossy Translation → LP → Invariant Detection pipeline.

    This is the main integration point. It:
    1. Runs NN inference on input data
    2. Translates NN outputs through lossy transformations
    3. Sets translated values as LP evidence
    4. Runs LP inference
    5. Feeds all observations to invariant detector / coordination controller
    6. Can generate proof trees on demand

    Usage:
        pipeline = CompositionPipeline(
            nn_component=nn_comp,
            lp_component=lp_comp,
            translation_config={
                "risk_drug_a": ("threshold", {"threshold": 0.5}),
                "risk_drug_b": ("threshold", {"threshold": 0.5}),
                "risk_interaction": ("categorical", {"boundaries": [0.3, 0.7]}),
            }
        )

        result = pipeline.run(patient_tensor)
        proof = pipeline.generate_proof()
    """

    def __init__(self,
                 nn_component: NNComponent,
                 lp_component: ProbLogComponent,
                 translation_config: Dict[str, Tuple[str, Dict]] = None,
                 persistence_threshold: float = 0.80):
        """
        Args:
            nn_component: The neural network component.
            lp_component: The logic program component (with rules already loaded).
            translation_config: Maps NN output name → (translation_type, params).
                translation_type is one of: "threshold", "quantize", "categorical",
                "soft_threshold", "modular".
                params are kwargs for the corresponding LossyTranslator method.
                If None, all outputs use threshold at 0.5.
            persistence_threshold: For invariant detection.
        """
        self.nn = nn_component
        self.lp = lp_component
        self.translation_config = translation_config or {}
        self.controller = CoordinationController(
            persistence_threshold=persistence_threshold,
            window_size=100
        )
        self.proof_gen = ProofGenerator()
        self.all_translation_records: List[List[TranslationRecord]] = []
        self.cycle_count = 0

    def run(self, input_tensor: torch.Tensor,
            additional_evidence: Dict[str, bool] = None) -> PipelineResult:
        """Run one complete pass through the composition pipeline.

        Args:
            input_tensor: Input data for NN (e.g., patient features).
            additional_evidence: Extra LP evidence not from NN
                (e.g., {"condition(diabetes)": True}).

        Returns:
            PipelineResult with all intermediate and final outputs.
        """
        self.cycle_count += 1

        # Step 1: NN inference
        nn_output = self.nn.predict(input_tensor)

        # Step 2: Lossy translation of NN outputs
        translated = {}
        records = []
        evidence = {}

        for name, value in nn_output.raw_outputs.items():
            record = self._translate(name, value)
            records.append(record)
            translated[name] = record.output_value

            # Map translated value to LP evidence atom
            evidence_atom = self._to_evidence_atom(name, record.output_value)
            if evidence_atom:
                evidence[evidence_atom] = record.output_value > 0.5

        # Add any additional evidence
        if additional_evidence:
            evidence.update(additional_evidence)

        self.all_translation_records.append(records)

        # Step 3: LP inference
        self.lp.set_evidence(evidence)
        query_atoms = self._default_queries()
        lp_output = self.lp.query(query_atoms)

        # Step 4: Feed observations to coordination controller
        # Combine NN outputs and LP results into one observation dict
        observation = {}
        for name, val in nn_output.raw_outputs.items():
            observation[f"nn_{name}"] = val
        for name, val in translated.items():
            observation[f"trans_{name}"] = val
        for atom, prob in lp_output.query_results.items():
            observation[f"lp_{atom}"] = prob

        events = self.controller.step(observation)

        return PipelineResult(
            nn_output=nn_output,
            translation_records=records,
            translated_values=translated,
            lp_output=lp_output,
            coordination_events=events
        )

    def generate_proof(self, system_name: str = "Medical Guidance System") -> ProofNode:
        """Generate a proof tree for currently detected invariants.

        Returns:
            ProofNode tree (render with proof.render()).
        """
        invariants = self.controller.get_stable_invariants()
        return self.proof_gen.generate_system_proof(
            invariants=invariants,
            translation_records=self.all_translation_records[-10:],  # Last 10 cycles
            system_name=system_name
        )

    def _translate(self, output_name: str, value: float) -> TranslationRecord:
        """Apply configured lossy translation to an NN output."""
        if output_name in self.translation_config:
            trans_type, params = self.translation_config[output_name]
        else:
            trans_type, params = "threshold", {"threshold": 0.5}

        translator = LossyTranslator
        if trans_type == "threshold":
            return translator.threshold(value, **params)
        elif trans_type == "quantize":
            return translator.quantize(value, **params)
        elif trans_type == "categorical":
            return translator.categorical_discretize(value, **params)
        elif trans_type == "soft_threshold":
            return translator.soft_threshold(value, **params)
        elif trans_type == "modular":
            return translator.modular_reduction(value, **params)
        else:
            return translator.threshold(value)

    def _to_evidence_atom(self, nn_output_name: str, translated_value: float) -> Optional[str]:
        """Convert NN output name + translated value to LP evidence atom.

        Convention: "risk_drug_a" → "high_efficacy(drug_a)"
        """
        clean = nn_output_name.replace("risk_", "")
        return f"high_efficacy({clean})"

    def _default_queries(self) -> List[str]:
        """Return default LP query atoms. Override in subclass for specific domains."""
        return ["safe(drug_a)", "safe(drug_b)", "approved(drug_a)", "approved(drug_b)"]
