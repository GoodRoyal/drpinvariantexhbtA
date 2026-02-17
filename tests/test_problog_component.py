import pytest
from composition.problog_component import ProbLogComponent, LPOutput


SIMPLE_RULES = """
0.9::approved(drug_a) :- high_efficacy(drug_a).
0.8::safe(drug_a) :- approved(drug_a).
"""


def test_query_returns_lpoutput():
    """query() should always return an LPOutput instance."""
    lp = ProbLogComponent()
    lp.load_rules(SIMPLE_RULES)
    lp.set_evidence({"high_efficacy(drug_a)": True})
    result = lp.query(["safe(drug_a)"])

    assert isinstance(result, LPOutput)
    assert "safe(drug_a)" in result.query_results


def test_evidence_true_raises_probability():
    """Providing positive evidence should produce non-zero probability."""
    lp = ProbLogComponent()
    lp.load_rules(SIMPLE_RULES)
    lp.set_evidence({"high_efficacy(drug_a)": True})
    result = lp.query(["safe(drug_a)", "approved(drug_a)"])

    assert result.query_results["approved(drug_a)"] > 0.0
    assert result.query_results["safe(drug_a)"] > 0.0


def test_no_evidence_gives_zero():
    """Without relevant evidence, derived predicates should be 0."""
    lp = ProbLogComponent()
    lp.load_rules(SIMPLE_RULES)
    lp.set_evidence({})
    result = lp.query(["safe(drug_a)", "approved(drug_a)"])

    assert result.query_results["approved(drug_a)"] == 0.0
    assert result.query_results["safe(drug_a)"] == 0.0


def test_add_rule_extends_program():
    """add_rule() should make new rules active in subsequent queries."""
    lp = ProbLogComponent()
    lp.load_rules("0.9::approved(drug_a) :- high_efficacy(drug_a).")
    lp.add_rule("0.8::safe(drug_a) :- approved(drug_a).")
    lp.set_evidence({"high_efficacy(drug_a)": True})

    result = lp.query(["safe(drug_a)"])
    assert result.query_results["safe(drug_a)"] > 0.0


def test_remove_rule_disables_rule():
    """remove_rule() should remove matching rules."""
    lp = ProbLogComponent()
    lp.load_rules(SIMPLE_RULES)
    removed = lp.remove_rule("safe(drug_a)")
    assert removed is True

    lp.set_evidence({"high_efficacy(drug_a)": True})
    result = lp.query(["safe(drug_a)"])
    assert result.query_results["safe(drug_a)"] == 0.0


def test_remove_nonexistent_rule_returns_false():
    """remove_rule() on missing fragment should return False."""
    lp = ProbLogComponent()
    lp.load_rules(SIMPLE_RULES)
    assert lp.remove_rule("no_such_predicate_xyz") is False


def test_chained_rules_forward_propagate():
    """Multi-hop rule chains should forward-chain through passes."""
    lp = ProbLogComponent()
    lp.load_rules("""
1.0::high_efficacy(drug_a) :- strong_trial(drug_a).
0.9::approved(drug_a) :- high_efficacy(drug_a).
0.8::safe(drug_a) :- approved(drug_a).
""")
    lp.set_evidence({"strong_trial(drug_a)": True})
    result = lp.query(["safe(drug_a)"])

    assert result.query_results["safe(drug_a)"] > 0.0


def test_query_atoms_not_in_rules_return_zero():
    """Queried atoms with no supporting rules should return 0."""
    lp = ProbLogComponent()
    lp.load_rules(SIMPLE_RULES)
    lp.set_evidence({})
    result = lp.query(["nonexistent(predicate)"])

    assert result.query_results["nonexistent(predicate)"] == 0.0


def test_proof_trace_populated_on_match():
    """proof_trace should contain entries when rules fire."""
    lp = ProbLogComponent()
    lp.load_rules(SIMPLE_RULES)
    lp.set_evidence({"high_efficacy(drug_a)": True})
    result = lp.query(["safe(drug_a)"])

    assert isinstance(result.proof_trace, list)
    assert isinstance(result.rules_fired, list)
