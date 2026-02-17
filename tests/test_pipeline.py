import pytest
import torch
from composition.nn_component import NNComponent, SimpleRiskNN
from composition.problog_component import ProbLogComponent
from composition.pipeline import CompositionPipeline, PipelineResult

RULES = """
0.9::approved(drug_a) :- high_efficacy(drug_a).
0.8::safe(drug_a) :- approved(drug_a).
0.7::approved(drug_b) :- high_efficacy(drug_b).
0.6::safe(drug_b) :- approved(drug_b).
"""


def make_pipeline(threshold=0.80):
    model = SimpleRiskNN(input_dim=4, output_dim=2)
    nn_comp = NNComponent(model, output_names=["risk_drug_a", "risk_drug_b"])
    lp_comp = ProbLogComponent()
    lp_comp.load_rules(RULES)
    return CompositionPipeline(
        nn_component=nn_comp,
        lp_component=lp_comp,
        translation_config={
            "risk_drug_a": ("threshold", {"threshold": 0.5}),
            "risk_drug_b": ("threshold", {"threshold": 0.5}),
        },
        persistence_threshold=threshold,
    )


def test_run_returns_pipeline_result():
    """A single run should return a fully populated PipelineResult."""
    pipeline = make_pipeline()
    result = pipeline.run(torch.zeros(4))

    assert isinstance(result, PipelineResult)
    assert len(result.translation_records) == 2
    assert "risk_drug_a" in result.translated_values
    assert "risk_drug_b" in result.translated_values


def test_translated_values_are_binary_after_threshold():
    """With threshold translation, translated values must be 0.0 or 1.0."""
    pipeline = make_pipeline()
    result = pipeline.run(torch.zeros(4))

    for val in result.translated_values.values():
        assert val in (0.0, 1.0)


def test_lp_output_contains_query_atoms():
    """LP query results should contain the default query atoms."""
    pipeline = make_pipeline()
    result = pipeline.run(torch.zeros(4))

    for atom in ["safe(drug_a)", "safe(drug_b)", "approved(drug_a)", "approved(drug_b)"]:
        assert atom in result.lp_output.query_results


def test_summary_is_string():
    """PipelineResult.summary() should return a non-empty string."""
    pipeline = make_pipeline()
    result = pipeline.run(torch.zeros(4))
    s = result.summary()
    assert isinstance(s, str)
    assert "Pipeline Result" in s


def test_cycle_count_increments():
    """cycle_count should increment with each run() call."""
    pipeline = make_pipeline()
    for i in range(5):
        pipeline.run(torch.randn(4))
    assert pipeline.cycle_count == 5


def test_translation_records_accumulate():
    """all_translation_records should grow with each run."""
    pipeline = make_pipeline()
    for _ in range(3):
        pipeline.run(torch.randn(4))
    assert len(pipeline.all_translation_records) == 3


def test_additional_evidence_passed_to_lp():
    """additional_evidence kwarg should be forwarded to LP."""
    pipeline = make_pipeline()
    # Inject direct evidence bypassing NN translation
    result = pipeline.run(torch.zeros(4), additional_evidence={"high_efficacy(drug_a)": True})
    assert result.lp_output.query_results.get("approved(drug_a)", 0.0) > 0.0


def test_generate_proof_returns_proof_node():
    """generate_proof() should return a ProofNode after enough observations."""
    from core.proof_generator import ProofNode
    pipeline = make_pipeline(threshold=0.70)
    for _ in range(5):
        pipeline.run(torch.zeros(4))
    proof = pipeline.generate_proof(system_name="Test")
    assert isinstance(proof, ProofNode)
    assert proof.level == 0


def test_all_translation_types_dispatch():
    """_translate should handle all five translation type strings."""
    model = SimpleRiskNN(input_dim=4, output_dim=5)
    nn_comp = NNComponent(model, output_names=["a", "b", "c", "d", "e"])
    lp_comp = ProbLogComponent()
    lp_comp.load_rules(RULES)
    pipeline = CompositionPipeline(
        nn_component=nn_comp,
        lp_component=lp_comp,
        translation_config={
            "a": ("threshold", {"threshold": 0.5}),
            "b": ("quantize", {"num_bins": 4}),
            "c": ("categorical", {"boundaries": [0.33, 0.66]}),
            "d": ("soft_threshold", {"threshold": 0.5}),
            "e": ("modular", {"modulus": 3}),
        }
    )
    result = pipeline.run(torch.rand(4))
    assert len(result.translation_records) == 5
