from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
from core.invariant_detector import InvariantDetector, Invariant


class CoordinationAction(Enum):
    MAINTAIN = "maintain"                          # Invariant stable, no action needed
    INCREASE_MONITORING = "increase_monitoring"    # Slight degradation
    FLAG_RECONFIGURATION = "flag_reconfiguration"  # Moderate degradation
    TRIGGER_FALLBACK = "trigger_fallback"          # Severe degradation
    INVALIDATE = "invalidate"                      # Invariant no longer holds


@dataclass
class CoordinationEvent:
    """Records a coordination action taken in response to invariant change."""
    cycle_number: int
    invariant_description: str
    action: CoordinationAction
    degradation: float          # 0.0 = no degradation, 1.0 = total loss
    message: str                # Human-readable explanation


class CoordinationController:
    """Monitors invariant persistence and triggers coordination responses.

    Implements the InvariantBasedCoordination algorithm from the patent:
    - degradation > 0.30 → invalidate + trigger_fallback
    - degradation > 0.20 → flag_reconfiguration
    - degradation > 0.10 → increase_monitoring
    - else → maintain_coordination

    Usage:
        controller = CoordinationController()

        # In each cycle:
        observations = {"nn_out": 0.82, "lp_pred": 1.0}
        events = controller.step(observations)

        for event in events:
            print(f"[{event.action.value}] {event.message}")
    """

    def __init__(self,
                 persistence_threshold: float = 0.80,
                 window_size: int = 100,
                 degradation_thresholds: Dict[str, float] = None):
        self.detector = InvariantDetector(
            persistence_threshold=persistence_threshold,
            window_size=window_size
        )
        self.baselines: Dict[str, float] = {}  # invariant_desc -> baseline persistence
        self.current_invariants: List[Invariant] = []
        self.cycle_count: int = 0
        self.event_log: List[CoordinationEvent] = []
        self.callbacks: Dict[CoordinationAction, List[Callable]] = {
            action: [] for action in CoordinationAction
        }

        # Patent-specified thresholds (can override)
        self.thresholds = degradation_thresholds or {
            "invalidate": 0.30,
            "reconfigure": 0.20,
            "monitor": 0.10,
        }

    def register_callback(self, action: CoordinationAction,
                          callback: Callable[[CoordinationEvent], None]) -> None:
        """Register a callback for a specific coordination action."""
        self.callbacks[action].append(callback)

    def step(self, observation: Dict[str, float]) -> List[CoordinationEvent]:
        """Process one interaction cycle.

        Args:
            observation: Maps agent_id -> translated scalar value for this cycle.

        Returns:
            List of coordination events triggered this cycle.
        """
        self.cycle_count += 1
        self.detector.observe(observation)

        events = []

        # Only start evaluating after enough observations
        if self.cycle_count < 20:
            return events

        # Re-detect invariants periodically (every 10 cycles)
        if self.cycle_count % 10 == 0:
            new_invariants = self.detector.detect_all()
            seen_keys = {inv.description for inv in new_invariants}

            # Update baselines for new invariants
            for inv in new_invariants:
                key = inv.description
                if key not in self.baselines:
                    self.baselines[key] = inv.persistence

            # Check degradation for known invariants still being detected
            for inv in new_invariants:
                key = inv.description
                if key in self.baselines:
                    baseline = self.baselines[key]
                    if baseline > 0:
                        degradation = (baseline - inv.persistence) / baseline
                        degradation = max(0.0, degradation)
                    else:
                        degradation = 0.0

                    action = self._determine_action(degradation)

                    if action != CoordinationAction.MAINTAIN:
                        event = CoordinationEvent(
                            cycle_number=self.cycle_count,
                            invariant_description=key,
                            action=action,
                            degradation=degradation,
                            message=self._format_message(inv, action, degradation)
                        )
                        events.append(event)
                        self.event_log.append(event)

                        # Fire callbacks
                        for cb in self.callbacks[action]:
                            cb(event)

            # Check for baseline invariants that have completely disappeared
            disappeared = [k for k in self.baselines if k not in seen_keys]
            for key in disappeared:
                degradation = 1.0
                action = self._determine_action(degradation)
                event = CoordinationEvent(
                    cycle_number=self.cycle_count,
                    invariant_description=key,
                    action=action,
                    degradation=degradation,
                    message=f"Invariant '{key}' disappeared — action: {action.value}"
                )
                events.append(event)
                self.event_log.append(event)
                for cb in self.callbacks[action]:
                    cb(event)
                del self.baselines[key]

            self.current_invariants = new_invariants

        return events

    def _determine_action(self, degradation: float) -> CoordinationAction:
        """Map degradation level to coordination action per patent algorithm."""
        if degradation > self.thresholds["invalidate"]:
            return CoordinationAction.TRIGGER_FALLBACK
        elif degradation > self.thresholds["reconfigure"]:
            return CoordinationAction.FLAG_RECONFIGURATION
        elif degradation > self.thresholds["monitor"]:
            return CoordinationAction.INCREASE_MONITORING
        else:
            return CoordinationAction.MAINTAIN

    def _format_message(self, invariant: Invariant,
                        action: CoordinationAction, degradation: float) -> str:
        """Generate human-readable coordination message."""
        return (
            f"Invariant [{invariant.invariant_type}] '{invariant.description}' — "
            f"degradation {degradation:.1%} — action: {action.value}"
        )

    def get_stable_invariants(self) -> List[Invariant]:
        """Return currently stable invariants (no significant degradation)."""
        return [inv for inv in self.current_invariants
                if inv.persistence >= self.detector.persistence_threshold]

    def get_event_log(self) -> List[CoordinationEvent]:
        """Return full history of coordination events."""
        return list(self.event_log)
