import pytest
from core.lossy_translation import LossyTranslator, TranslationChain


def test_quantize_lossy():
    """Different inputs should map to same output (demonstrating loss)."""
    r1 = LossyTranslator.quantize(0.13, num_bins=4)
    r2 = LossyTranslator.quantize(0.24, num_bins=4)
    assert r1.output_value == r2.output_value  # Both in bin 0
    assert r1.input_value != r2.input_value     # But different inputs


def test_threshold_binary():
    """Threshold should produce exactly 0.0 or 1.0."""
    assert LossyTranslator.threshold(0.7).output_value == 1.0
    assert LossyTranslator.threshold(0.3).output_value == 0.0
    assert LossyTranslator.threshold(0.5).output_value == 1.0  # >= threshold


def test_modular_reduction():
    """Modular reduction should be lossy (5 and 1 both → 1 mod 4)."""
    r1 = LossyTranslator.modular_reduction(5.0, modulus=4)
    r2 = LossyTranslator.modular_reduction(1.0, modulus=4)
    assert r1.output_value == r2.output_value == 1.0


def test_categorical_discretize():
    """Should map to correct categories."""
    assert LossyTranslator.categorical_discretize(0.1).output_value == 0.0   # LOW
    assert LossyTranslator.categorical_discretize(0.5).output_value == 1.0   # MEDIUM
    assert LossyTranslator.categorical_discretize(0.9).output_value == 2.0   # HIGH


def test_translation_chain():
    """Chain should apply steps sequentially and record all losses."""
    chain = TranslationChain()
    chain.add_step("quantize", lambda v: LossyTranslator.quantize(v, num_bins=4))
    chain.add_step("threshold", lambda v: LossyTranslator.threshold(v, threshold=0.5))

    final, records = chain.translate(0.73)
    assert len(records) == 2
    assert records[0].source_space == "continuous"
    assert isinstance(final, float)
    assert final in (0.0, 1.0)  # After threshold, must be binary
