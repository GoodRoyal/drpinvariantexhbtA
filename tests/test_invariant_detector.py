import pytest
from core.invariant_detector import InvariantDetector, Invariant


def test_ordering_invariant_detected():
    """Agent A consistently > Agent B should produce ordering invariant."""
    detector = InvariantDetector(persistence_threshold=0.80, window_size=50)

    for i in range(50):
        detector.observe({
            "agent_a": 0.8 + 0.1 * (i % 3) / 3,  # Always around 0.8-0.9
            "agent_b": 0.2 + 0.1 * (i % 5) / 5,  # Always around 0.2-0.3
        })

    invariants = detector.detect_ordering_invariants()
    assert len(invariants) >= 1
    ordering = invariants[0]
    assert ordering.invariant_type == "ordering"
    assert ordering.persistence >= 0.80
    assert "agent_a" in ordering.agents_involved
    assert "agent_b" in ordering.agents_involved


def test_no_ordering_when_random():
    """Random values should not produce ordering invariant."""
    import numpy as np
    np.random.seed(42)
    detector = InvariantDetector(persistence_threshold=0.80, window_size=100)

    for _ in range(100):
        detector.observe({
            "agent_a": np.random.random(),
            "agent_b": np.random.random(),
        })

    invariants = detector.detect_ordering_invariants()
    # Should be empty or have low persistence
    high_persistence = [inv for inv in invariants if inv.persistence >= 0.80]
    assert len(high_persistence) == 0


def test_bounded_oscillation_detected():
    """Agent with small variance should produce bounded oscillation."""
    import numpy as np
    detector = InvariantDetector(persistence_threshold=0.70, window_size=50)

    for i in range(50):
        detector.observe({
            "stable_agent": 0.5 + 0.02 * np.sin(i / 5),  # Tight oscillation
            "wild_agent": np.random.random(),               # All over the place
        })

    invariants = detector.detect_bounded_oscillation()
    stable_invs = [inv for inv in invariants if "stable_agent" in inv.agents_involved]
    assert len(stable_invs) >= 1


def test_recurrence_detected():
    """Repeated pattern should be detected."""
    detector = InvariantDetector(persistence_threshold=0.30, window_size=60)

    # Create a repeating pattern: H, L, M, H, L, M, ...
    pattern_values = [0.9, 0.1, 0.5]  # Maps to H, L, M
    for i in range(60):
        detector.observe({
            "patterned": pattern_values[i % 3],
        })

    invariants = detector.detect_recurrence_patterns()
    assert len(invariants) >= 1


def test_detect_all_returns_mixed_types():
    """detect_all should return invariants of multiple types."""
    import numpy as np
    detector = InvariantDetector(persistence_threshold=0.70, window_size=50)

    for i in range(50):
        detector.observe({
            "high_agent": 0.8 + 0.05 * np.sin(i / 5),
            "low_agent": 0.2 + 0.05 * np.cos(i / 5),
        })

    invariants = detector.detect_all()
    types_found = {inv.invariant_type for inv in invariants}
    # Should find at least ordering (high > low consistently)
    assert "ordering" in types_found
