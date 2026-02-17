import pytest
import numpy as np
from core.coordination_controller import CoordinationController, CoordinationAction


def test_stable_system_no_events():
    """A stable system should not trigger coordination events."""
    ctrl = CoordinationController(persistence_threshold=0.80, window_size=50)

    all_events = []
    for i in range(60):
        events = ctrl.step({"a": 0.8, "b": 0.2})
        all_events.extend(events)

    # Stable ordering a > b — no degradation events expected
    fallbacks = [e for e in all_events if e.action == CoordinationAction.TRIGGER_FALLBACK]
    assert len(fallbacks) == 0


def test_degradation_triggers_event():
    """A sudden change in ordering should trigger coordination response."""
    ctrl = CoordinationController(persistence_threshold=0.70, window_size=30)

    # Phase 1: establish baseline (a > b)
    for i in range(40):
        ctrl.step({"a": 0.9, "b": 0.1})

    # Phase 2: disrupt (now b > a)
    all_events = []
    for i in range(30):
        events = ctrl.step({"a": 0.1, "b": 0.9})
        all_events.extend(events)

    # Should have triggered at least one non-MAINTAIN event
    assert len(all_events) > 0
    actions = {e.action for e in all_events}
    assert CoordinationAction.MAINTAIN not in actions
