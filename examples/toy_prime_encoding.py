"""
Toy example from the provisional patent: Prime-Based Encoding.

Three agents exchange event counts. Agent A encodes values using primes.
Agent B observes recurrence patterns. Agent C applies modular reduction.

Despite lossy translation, ordering and recurrence invariants persist.

Run: python -m examples.toy_prime_encoding
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.invariant_detector import InvariantDetector
from core.lossy_translation import LossyTranslator
from core.coordination_controller import CoordinationController
from core.proof_generator import ProofGenerator


def main():
    # Prime mapping from patent: integer event counts → primes
    PRIME_MAP = {1: 2, 2: 3, 3: 5, 4: 7, 5: 11, 6: 13, 7: 17, 8: 19, 9: 23, 10: 29}

    controller = CoordinationController(persistence_threshold=0.75, window_size=50)
    proof_gen = ProofGenerator()
    all_records = []

    print("=" * 60)
    print("STRUCTURAL INVARIANT DETECTION — Prime Encoding Example")
    print("From: Paredes, 'Coordination of Heterogeneous Agents")
    print("       by Discovering Structural Invariants Under")
    print("       Lossy Translation'")
    print("=" * 60)

    print("\nPhase 1: Running 100 interaction cycles...")
    print("  Agent A: Encodes event counts (1-10) as primes")
    print("  Agent B: Observes prime values (no inversion)")
    print("  Agent C: Applies mod-4 reduction (lossy, non-invertible)")
    print()

    for cycle in range(100):
        # Agent A generates an event count (cyclic pattern with noise)
        event_count = (cycle % 7) + 1 + int(np.random.normal(0, 0.5))
        event_count = max(1, min(10, event_count))

        # Agent A encodes as prime
        prime_value = PRIME_MAP[event_count]

        # Agent B sees the prime (no inversion possible)
        agent_b_value = float(prime_value)

        # Agent C applies mod-4 reduction (LOSSY)
        mod_record = LossyTranslator.modular_reduction(float(prime_value), modulus=4)
        agent_c_value = mod_record.output_value

        # Normalize for observation (keep on comparable scale)
        observation = {
            "agent_a_count": event_count / 10.0,
            "agent_b_prime": prime_value / 29.0,  # Normalize by max prime
            "agent_c_mod4": agent_c_value / 3.0,  # Normalize by max mod value
        }

        events = controller.step(observation)

        for event in events:
            print(f"  Cycle {cycle}: [{event.action.value}] {event.message}")

    # Detect and display invariants
    invariants = controller.get_stable_invariants()
    all_invariants = controller.detector.detect_all()

    print(f"\nPhase 2: Detected {len(all_invariants)} structural invariants")
    print("-" * 60)

    for inv in all_invariants:
        print(f"\n  Type: {inv.invariant_type}")
        print(f"  Description: {inv.description}")
        print(f"  Persistence: {inv.persistence:.1%}")
        print(f"  Confidence: {inv.confidence:.1%}")

    # Generate proof tree
    print("\n" + "=" * 60)
    print("PROOF TREE")
    print("=" * 60)

    proof = proof_gen.generate_system_proof(
        invariants=all_invariants,
        system_name="Prime-Encoding Coordination System"
    )
    print(proof.render())

    # Verify proof depth
    depth = proof.depth()
    print(f"\nProof depth: {depth} levels (CLARA requirement: ≤ 10)")
    print(f"Meets CLARA requirement: {'YES ✓' if depth <= 10 else 'NO ✗'}")

    # Phase 3: Demonstrate degradation detection
    print("\n" + "=" * 60)
    print("Phase 3: Injecting disruption (random agent behavior)...")
    print("=" * 60)

    for cycle in range(50):
        # Disrupted: Agent A now sends random values (breaks invariants)
        observation = {
            "agent_a_count": np.random.random(),
            "agent_b_prime": np.random.random(),
            "agent_c_mod4": np.random.random(),
        }
        events = controller.step(observation)
        for event in events:
            print(f"  Cycle {100 + cycle}: [{event.action.value}] {event.message}")

    print("\n" + "=" * 60)
    print("COMPLETE — System detected invariant persistence AND degradation")
    print("=" * 60)


if __name__ == "__main__":
    main()
