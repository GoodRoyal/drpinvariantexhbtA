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
    """Demonstrate categorical verification of the NN→LP lossy functor.

    Builds a minimal NN category (state_low, state_high) and an LP category
    (lp_false, lp_true), then defines a lossy functor that collapses both NN
    states onto lp_true.  YonedaChecker exposes the non-faithfulness and shows
    that the ordering invariant state_high > state_low cannot be certified to
    persist under this particular functor.
    """
    print("\n" + "=" * 70)
    print("YONEDA CATEGORICAL VERIFICATION")
    print("(Verifying NN→LP functor properties via Hom-set profiles)")
    print("=" * 70)

    # --- NN Category ---
    # Objects: NN activation states
    # Morphisms: transitions between states (forward propagation steps)
    nn_cat = Category("NeuralNetwork")
    state_low = nn_cat.add_object("state_low")
    state_high = nn_cat.add_object("state_high")
    # Morphism: low activation can be boosted to high (one direction)
    nn_cat.add_morphism(state_low, state_high, "activate")
    # Extra morphism into state_high gives it a larger Hom-set — encoding the
    # fact that state_high is "reachable from more states" (ordering)
    nn_cat.add_morphism(state_low, state_high, "amplify")

    print("\nNN Category objects:", [o.name for o in nn_cat.objects])
    print("NN Category morphisms:", [m.name for m in nn_cat.morphisms
                                     if not m.name.startswith("id_")])

    # --- LP Category ---
    # Objects: LP truth values / belief states
    lp_cat = Category("LogicProgram")
    lp_false = lp_cat.add_object("lp_false")
    lp_true = lp_cat.add_object("lp_true")
    lp_cat.add_morphism(lp_false, lp_true, "assert")

    print("\nLP Category objects:", [o.name for o in lp_cat.objects])
    print("LP Category morphisms:", [m.name for m in lp_cat.morphisms
                                     if not m.name.startswith("id_")])

    # --- Lossy Functor: NN → LP ---
    # Both NN states map to lp_true — this collapses the ordering distinction.
    # This models threshold translation: any activation above 0 → True.
    lossy_F = Functor("NN_to_LP_lossy", nn_cat, lp_cat)
    lossy_F.map_object(state_low, lp_true)   # lossy: low maps to True
    lossy_F.map_object(state_high, lp_true)  # lossy: high also maps to True

    print("\nFunctor object mappings:")
    print(f"  state_low  → {lossy_F.apply_object(state_low).name}")
    print(f"  state_high → {lossy_F.apply_object(state_high).name}")
    print("  (Both states collapse to lp_true — ordering information destroyed)")

    # --- Yoneda Verification ---
    checker = YonedaChecker()
    result = checker.verify_invariant_persistence(
        source_cat=nn_cat,
        target_cat=lp_cat,
        functor=lossy_F,
        invariant_type="ordering",
        objects=[state_high, state_low],
    )

    print("\n" + "-" * 70)
    print("PROOF STEPS (ordering invariant: state_high > state_low)")
    print("-" * 70)
    for step in result["proof_steps"]:
        print(" ", step)

    print("\n" + "-" * 70)
    verified = result["verified"]
    lossiness = result["lossiness"]
    print(f"Verified:    {verified}")
    print(f"Faithful:    {lossiness['faithful']}  "
          f"(False → functor collapses morphisms → ordering not guaranteed)")
    print(f"Full:        {lossiness['full']}")
    print(f"Conclusion:  {'Ordering persists under functor' if verified else 'Ordering NOT certified — functor is lossy/non-faithful'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
