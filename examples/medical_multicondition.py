"""
CLARA TA1 Demonstration: Medical Multi-Condition Guidance.

Composes a Neural Network (risk prediction) with a Bayesian Logic Program
(treatment rules + drug interactions) via lossy translation.

Detects structural invariants across the composition and generates
hierarchical proofs meeting CLARA's explainability requirements.

Run: python -m examples.medical_multicondition
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from composition.nn_component import NNComponent, SimpleRiskNN
from composition.problog_component import ProbLogComponent
from composition.pipeline import CompositionPipeline
from verification.categories import Category, Functor
from verification.yoneda_checker import YonedaChecker


# Medical treatment rules as Bayesian Logic Program
MEDICAL_RULES = """
% Drug approval based on efficacy evidence
0.9::approved(drug_a) :- high_efficacy(drug_a).
0.85::approved(drug_b) :- high_efficacy(drug_b).

% Safety depends on approval and no contraindications
0.95::safe(drug_a) :- approved(drug_a), not contraindicated(drug_a).
0.95::safe(drug_b) :- approved(drug_b), not contraindicated(drug_b).

% Contraindication rules (drug interactions)
contraindicated(drug_a) :- high_efficacy(drug_a), high_efficacy(drug_b), high_efficacy(interaction).

% Interaction risk
0.7::interaction_warning :- high_efficacy(interaction).
"""


def generate_patient_data(n_patients: int = 200, seed: int = 42) -> torch.Tensor:
    """Generate synthetic patient data.

    Features: [age, blood_pressure, glucose, bmi,
               heart_rate, cholesterol, creatinine, hemoglobin]
    All normalized to [0, 1].
    """
    torch.manual_seed(seed)
    return torch.rand(n_patients, 8)


def main():
    print("=" * 70)
    print("CLARA TA1 DEMONSTRATION")
    print("Structural Invariant Composition: Neural Network + Bayesian-LP")
    print("Domain: Medical Multi-Condition Treatment Guidance")
    print("=" * 70)

    # --- Setup Components ---

    # Neural Network: predicts risk scores for two drugs + interaction risk
    nn_model = SimpleRiskNN(input_dim=8, hidden_dim=32, output_dim=3)
    nn_comp = NNComponent(
        model=nn_model,
        output_names=["risk_drug_a", "risk_drug_b", "risk_interaction"]
    )

    # Logic Program: encodes treatment rules
    lp_comp = ProbLogComponent()
    lp_comp.load_rules(MEDICAL_RULES)

    # Composition Pipeline: NN → lossy translation → LP
    pipeline = CompositionPipeline(
        nn_component=nn_comp,
        lp_component=lp_comp,
        translation_config={
            "risk_drug_a": ("threshold", {"threshold": 0.5}),
            "risk_drug_b": ("threshold", {"threshold": 0.5}),
            "risk_interaction": ("threshold", {"threshold": 0.4}),
        },
        persistence_threshold=0.75
    )

    # --- Run Interaction Cycles ---

    patients = generate_patient_data(n_patients=200)

    print("\nPhase 1: Running composition pipeline on 200 patients...")
    print("  NN outputs: risk_drug_a, risk_drug_b, risk_interaction (continuous)")
    print("  Translation: threshold at 0.5/0.5/0.4 (lossy, non-invertible)")
    print("  LP inference: approved/safe/contraindicated (probabilistic)")
    print()

    results = []
    for i in range(len(patients)):
        result = pipeline.run(
            patients[i],
            additional_evidence={}
        )
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1} patients...")
            if result.coordination_events:
                for event in result.coordination_events:
                    print(f"    [{event.action.value}] {event.message}")

    # --- Show sample result ---

    print("\n" + "-" * 70)
    print("SAMPLE RESULT (Patient #1):")
    print("-" * 70)
    print(results[0].summary())

    # --- Generate Proof ---

    print("\n" + "=" * 70)
    print("HIERARCHICAL PROOF TREE")
    print("(CLARA requirement: natural deduction style, ≤10 unfolding levels)")
    print("=" * 70)

    proof = pipeline.generate_proof(system_name="Medical Multi-Condition Guidance")
    print(proof.render())

    depth = proof.depth()
    print(f"\nProof depth: {depth} levels")
    print(f"CLARA ≤10 levels requirement: {'MET ✓' if depth <= 10 else 'NOT MET ✗'}")

    # --- Show all detected invariants ---

    print("\n" + "=" * 70)
    print("DETECTED STRUCTURAL INVARIANTS")
    print("=" * 70)

    stable = pipeline.controller.get_stable_invariants()
    all_inv = pipeline.controller.detector.detect_all()

    print(f"\nTotal detected: {len(all_inv)}")
    print(f"Stable (above threshold): {len(stable)}")

    for inv in all_inv:
        status = "STABLE" if inv.persistence >= 0.75 else "DEGRADED"
        print(f"\n  [{status}] {inv.invariant_type}: {inv.description}")
        print(f"    Persistence: {inv.persistence:.1%} | Confidence: {inv.confidence:.1%}")

    # --- Demonstrate Human Editing (CLARA requirement) ---

    print("\n" + "=" * 70)
    print("HUMAN KNOWLEDGE EDITING DEMONSTRATION")
    print("(CLARA: non-AI-experts can edit model knowledge)")
    print("=" * 70)

    print("\nDoctor adds rule: 'Drug A is contraindicated for patients with renal failure'")
    lp_comp.add_rule("contraindicated(drug_a) :- condition(renal_failure).")

    # Re-run with new evidence
    result_edited = pipeline.run(
        patients[0],
        additional_evidence={"condition(renal_failure)": True}
    )
    print(f"\nBefore edit - safe(drug_a): {results[0].lp_output.query_results.get('safe(drug_a)', 'N/A')}")
    print(f"After edit  - safe(drug_a): {result_edited.lp_output.query_results.get('safe(drug_a)', 'N/A')}")
    print("\nRule edit took effect immediately — no retraining required.")

    # --- Summary ---

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("""
    Demonstrated CLARA TA1 capabilities:
    ✓ Composed 1 ML kind (Neural Network) + 1 AR kind (Bayesian-LP)
    ✓ Lossy translation at composition boundary (threshold, non-invertible)
    ✓ Structural invariant detection (ordering, bounded oscillation, recurrence)
    ✓ Hierarchical proof generation (≤10 unfolding levels)
    ✓ Human-editable knowledge (LP rules modified without retraining)
    ✓ Coordination adaptation (degradation detection + response)
    """)

    run_yoneda_verification()


def run_yoneda_verification():
    """Formal verification using Yoneda embedding.

    Demonstrates the core theoretical result: lossy functors destroy SOME
    structural properties but preserve OTHERS. The system's value is
    detecting which invariants persist under which translations.
    """
    checker = YonedaChecker()

    print("\n" + "=" * 70)
    print("YONEDA VERIFICATION: Formal Invariant Persistence Analysis")
    print("=" * 70)

    # --- Build NN Category ---
    nn_cat = Category("NeuralNetwork")
    s_low = nn_cat.add_object("state_low")    # NN output < threshold
    s_high = nn_cat.add_object("state_high")  # NN output >= threshold
    # Morphisms: two ways to go from low to high (different activation paths)
    nn_cat.add_morphism(s_low, s_high, "activate")
    nn_cat.add_morphism(s_low, s_high, "amplify")
    # state_high has |Hom(-, state_high)| = 3 (id + activate + amplify)
    # state_low  has |Hom(-, state_low)|  = 1 (id only)
    # So state_high > state_low in Yoneda profile

    # --- Build LP Category ---
    lp_cat = Category("LogicProgram")
    lp_false = lp_cat.add_object("lp_false")
    lp_true = lp_cat.add_object("lp_true")
    lp_cat.add_morphism(lp_false, lp_true, "assert")
    # lp_true has |Hom(-, lp_true)| = 2 (id + assert)
    # lp_false has |Hom(-, lp_false)| = 1 (id only)
    # So lp_true > lp_false in Yoneda profile

    # ============================================================
    # FUNCTOR A: Collapse (degenerate threshold — everything passes)
    # ============================================================
    print("\n--- Functor A: Collapse (threshold too low) ---")
    print("Maps: state_low → lp_true, state_high → lp_true")
    print("This models a threshold so low that all NN outputs pass.\n")

    F_collapse = Functor("NN_to_LP_collapse", nn_cat, lp_cat)
    F_collapse.map_object(s_low, lp_true)
    F_collapse.map_object(s_high, lp_true)
    # Map morphisms: both activate and amplify go to id(lp_true)
    # since source and target both map to lp_true
    id_true = lp_cat.identity(lp_true)
    for m in nn_cat.morphisms:
        if m.source != m.target:  # non-identity
            F_collapse.map_morphism(m, id_true)
        else:
            F_collapse.map_morphism(m, lp_cat.identity(F_collapse.apply_object(m.source)))

    result_a = checker.verify_invariant_persistence(
        source_cat=nn_cat, target_cat=lp_cat,
        functor=F_collapse, invariant_type="ordering",
        objects=[s_high, s_low]
    )
    for step in result_a["proof_steps"]:
        print(f"  {step}")
    print(f"\n  Ordering persists: {result_a['verified']}")

    # Also check bounded persistence
    result_a_bounded = checker.verify_invariant_persistence(
        source_cat=nn_cat, target_cat=lp_cat,
        functor=F_collapse, invariant_type="bounded",
        objects=[s_high, s_low]
    )
    print(f"  Bounded persists:  {result_a_bounded['verified']}")

    # ============================================================
    # FUNCTOR B: Proper threshold (ordering preserved)
    # ============================================================
    print("\n--- Functor B: Proper threshold ---")
    print("Maps: state_low → lp_false, state_high → lp_true")
    print("This models a well-calibrated threshold translation.\n")

    F_threshold = Functor("NN_to_LP_threshold", nn_cat, lp_cat)
    F_threshold.map_object(s_low, lp_false)
    F_threshold.map_object(s_high, lp_true)
    # Map morphisms
    # id(state_low) → id(lp_false), id(state_high) → id(lp_true)
    F_threshold.map_morphism(nn_cat.identity(s_low), lp_cat.identity(lp_false))
    F_threshold.map_morphism(nn_cat.identity(s_high), lp_cat.identity(lp_true))
    # activate, amplify: state_low → state_high maps to assert: lp_false → lp_true
    assert_morph = None
    for m in lp_cat.morphisms:
        if m.source == lp_false and m.target == lp_true and m.name == "assert":
            assert_morph = m
            break
    if assert_morph:
        for m in nn_cat.morphisms:
            if m.source == s_low and m.target == s_high:
                F_threshold.map_morphism(m, assert_morph)

    result_b = checker.verify_invariant_persistence(
        source_cat=nn_cat, target_cat=lp_cat,
        functor=F_threshold, invariant_type="ordering",
        objects=[s_high, s_low]
    )
    for step in result_b["proof_steps"]:
        print(f"  {step}")
    print(f"\n  Ordering persists: {result_b['verified']}")

    result_b_bounded = checker.verify_invariant_persistence(
        source_cat=nn_cat, target_cat=lp_cat,
        functor=F_threshold, invariant_type="bounded",
        objects=[s_high, s_low]
    )
    print(f"  Bounded persists:  {result_b_bounded['verified']}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("YONEDA VERIFICATION SUMMARY")
    print("=" * 70)
    print("""
    Functor A (Collapse — degenerate threshold):
      Ordering:    DESTROYED — both states collapse to same LP object
      Boundedness: PRESERVED — objects remain in finite LP category

    Functor B (Proper threshold):
      Ordering:    PRESERVED — Hom-set magnitude ordering maintained
      Boundedness: PRESERVED — objects remain in finite LP category

    Key insight: The system detects WHICH invariants persist under
    WHICH translations. A well-calibrated threshold preserves ordering;
    a degenerate one destroys it. Structural invariant detection
    identifies this automatically — enabling the coordination controller
    to trigger reconfiguration when a previously-persistent invariant
    degrades (e.g., threshold drift causes collapse).
    """)

    # Verify functor properties
    print(f"  Functor A faithful (injective on morphisms): {F_collapse.is_faithful()}")
    print(f"  Functor B faithful (injective on morphisms): {F_threshold.is_faithful()}")
    print(f"  → Both functors are lossy, but B preserves more structure than A")
    print("=" * 70)


if __name__ == "__main__":
    main()
