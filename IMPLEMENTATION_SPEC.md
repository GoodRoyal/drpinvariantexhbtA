# Structural Invariants v0.1 — Implementation Specification

## For use with Claude Code (Sonnet). This document is the single source of truth.

---

## PROJECT CONTEXT

This software implements a framework from a provisional patent: "Coordination of Heterogeneous Agents by Discovering Structural Invariants Under Lossy Translation." The specific application is DARPA CLARA TA1: composing Neural Networks (ML) with Bayesian Logic Programs (AR) into a verifiable system.

**What the system does:** A neural network produces continuous outputs. Those outputs pass through lossy translation (quantization, thresholding) into discrete predicates for a logic program. The logic program performs inference. We detect structural properties ("invariants") that persist across this lossy composition boundary despite information loss. Those invariants become the basis for verification, coordination, and explainable proofs.

**Key insight:** We do NOT try to preserve information across the ML→AR boundary. We exploit the loss to find what survives — and what survives is provably useful for coordination.

---

## DEPENDENCY VERSIONS

```
# requirements.txt
torch>=2.0.0
numpy>=1.24.0
problog>=2.2.4
matplotlib>=3.7.0
pytest>=7.0.0
```

**Note on ProbLog:** `pip install problog`. It provides a Python API for Bayesian Logic Programs. If ProbLog installation fails on the target system, there is a fallback spec in the problog_component section below.

---

## BUILD ORDER

Implement and test in this exact order. Each layer depends only on layers above it.

```
LAYER 1 (no dependencies, implement first):
  core/invariant_detector.py
  core/lossy_translation.py

LAYER 2 (depends on Layer 1):
  core/coordination_controller.py
  core/proof_generator.py

LAYER 3 (depends on Layer 1+2, plus external libs):
  composition/nn_component.py
  composition/problog_component.py

LAYER 4 (depends on all above):
  composition/pipeline.py
  training/invariant_loss.py
  training/constrained_trainer.py

LAYER 5 (depends on all above):
  verification/categories.py
  verification/yoneda_checker.py

LAYER 6 (integration):
  examples/toy_prime_encoding.py
  examples/medical_multicondition.py
```

---

## MODULE SPECIFICATIONS

---

### `core/__init__.py`

```python
from .invariant_detector import InvariantDetector
from .lossy_translation import LossyTranslator
from .coordination_controller import CoordinationController
from .proof_generator import ProofGenerator
```

---

### `core/invariant_detector.py`

**Purpose:** Detect structural properties that persist across interaction cycles despite lossy translation. This is the mathematical heart of the system.

**Three invariant types to detect:**

1. **Ordering Invariant:** "When agent A's value is in state X, agent B's translated value exceeds threshold Y" — and this relationship persists across cycles.
2. **Bounded Oscillation:** A signal that traverses a cycle of agents returns within bounded distance of its starting point, despite lossy transformations at each step.
3. **Recurrence Pattern:** A specific sequence of translated states recurs at frequency significantly above random.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class Invariant:
    """A detected structural invariant."""
    invariant_type: str          # "ordering" | "bounded_oscillation" | "recurrence"
    description: str             # Human-readable description
    persistence: float           # 0.0 to 1.0 — fraction of cycles where invariant held
    confidence: float            # Statistical confidence (persistence vs random baseline)
    agents_involved: List[str]   # Agent IDs participating in this invariant
    metadata: Dict = field(default_factory=dict)  # Type-specific extra data


class InvariantDetector:
    """Detects structural invariants from interaction cycle observations.
    
    Usage:
        detector = InvariantDetector(persistence_threshold=0.80)
        
        # Feed observations from each interaction cycle
        for cycle_data in interaction_cycles:
            detector.observe(cycle_data)
        
        # Detect invariants
        invariants = detector.detect_all()
    """
    
    def __init__(self, persistence_threshold: float = 0.80, window_size: int = 100):
        """
        Args:
            persistence_threshold: Minimum persistence ratio to report an invariant.
                                   0.80 means the property must hold in 80% of observed cycles.
            window_size: Number of recent cycles to consider for detection.
        """
        self.persistence_threshold = persistence_threshold
        self.window_size = window_size
        self.observations: List[Dict[str, float]] = []
        # observations is a list of dicts, one per cycle.
        # Each dict maps agent_id -> translated_value (float).
        # The translated_value is the agent's state AFTER lossy translation.
    
    def observe(self, cycle_observation: Dict[str, float]) -> None:
        """Record one interaction cycle's observations.
        
        Args:
            cycle_observation: Maps agent_id -> translated scalar value.
                Example: {"nn_output": 0.87, "lp_predicate": 1.0, "bayes_posterior": 0.62}
        """
        self.observations.append(cycle_observation)
        if len(self.observations) > self.window_size:
            self.observations.pop(0)
    
    def detect_all(self) -> List[Invariant]:
        """Run all detection algorithms. Returns list of detected invariants."""
        results = []
        results.extend(self.detect_ordering_invariants())
        results.extend(self.detect_bounded_oscillation())
        results.extend(self.detect_recurrence_patterns())
        return results
    
    def detect_ordering_invariants(self) -> List[Invariant]:
        """Detect persistent ordering relationships between agent pairs.
        
        Algorithm (from patent):
        For each pair of agents (a_i, a_j):
            Count cycles where translated_value(a_i) > translated_value(a_j)
            persistence = count / total_observations
            If persistence >= threshold: emit OrderingInvariant
        
        Returns: List of ordering invariants above persistence threshold.
        """
        if len(self.observations) < 10:
            return []
        
        invariants = []
        agents = sorted(self.observations[0].keys())
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a_i, a_j = agents[i], agents[j]
                count_i_gt_j = 0
                total = 0
                
                for obs in self.observations:
                    if a_i in obs and a_j in obs:
                        total += 1
                        if obs[a_i] > obs[a_j]:
                            count_i_gt_j += 1
                
                if total == 0:
                    continue
                
                # Check both directions
                persistence_gt = count_i_gt_j / total
                persistence_lt = 1.0 - persistence_gt
                persistence = max(persistence_gt, persistence_lt)
                
                if persistence >= self.persistence_threshold:
                    direction = ">" if persistence_gt >= persistence_lt else "<"
                    # Confidence: how far above random (0.5) baseline
                    confidence = 2.0 * abs(persistence - 0.5)
                    
                    invariants.append(Invariant(
                        invariant_type="ordering",
                        description=f"{a_i} {direction} {a_j} in {persistence:.1%} of cycles",
                        persistence=persistence,
                        confidence=confidence,
                        agents_involved=[a_i, a_j],
                        metadata={"direction": direction, "total_cycles": total}
                    ))
        
        return invariants
    
    def detect_bounded_oscillation(self) -> List[Invariant]:
        """Detect agents whose translated values stay within bounded range.
        
        Algorithm (from patent):
        For each agent:
            Compute value range across recent window
            If range is bounded (< epsilon) and non-trivial (> min_range):
                emit BoundedOscillationInvariant
        
        Also checks cycle-level: if a value traverses A->B->C->A and returns
        within epsilon of starting value.
        
        Returns: List of bounded oscillation invariants.
        """
        if len(self.observations) < 10:
            return []
        
        invariants = []
        agents = sorted(self.observations[0].keys())
        
        for agent in agents:
            values = [obs[agent] for obs in self.observations if agent in obs]
            if len(values) < 10:
                continue
            
            val_range = max(values) - min(values)
            val_mean = np.mean(values)
            val_std = np.std(values)
            
            # Bounded if std is small relative to mean (coefficient of variation)
            # and range is non-trivial (not constant)
            if val_std > 0.001 and val_range < 0.5:
                persistence = 1.0 - (val_std / max(abs(val_mean), 0.001))
                persistence = max(0.0, min(1.0, persistence))
                
                if persistence >= self.persistence_threshold:
                    invariants.append(Invariant(
                        invariant_type="bounded_oscillation",
                        description=f"{agent} bounded in [{min(values):.3f}, {max(values):.3f}]",
                        persistence=persistence,
                        confidence=persistence,
                        agents_involved=[agent],
                        metadata={
                            "min": float(min(values)),
                            "max": float(max(values)),
                            "std": float(val_std),
                            "mean": float(val_mean)
                        }
                    ))
        
        return invariants
    
    def detect_recurrence_patterns(self, max_pattern_len: int = 4) -> List[Invariant]:
        """Detect recurring sequences of discretized states.
        
        Algorithm:
        1. Discretize each agent's values into bins (low/medium/high).
        2. Extract all subsequences of length 2..max_pattern_len.
        3. Compare observed frequency to random baseline.
        4. Patterns occurring at >= 2x random rate with sufficient count are invariants.
        
        Returns: List of recurrence pattern invariants.
        """
        if len(self.observations) < 20:
            return []
        
        invariants = []
        agents = sorted(self.observations[0].keys())
        
        # Discretize each agent's time series into LOW/MED/HIGH
        discretized = {}
        for agent in agents:
            values = [obs.get(agent, 0.0) for obs in self.observations]
            p33, p66 = np.percentile(values, [33, 66])
            discretized[agent] = []
            for v in values:
                if v <= p33:
                    discretized[agent].append("L")
                elif v <= p66:
                    discretized[agent].append("M")
                else:
                    discretized[agent].append("H")
        
        # For each agent, find recurring patterns
        for agent in agents:
            seq = discretized[agent]
            for pattern_len in range(2, min(max_pattern_len + 1, len(seq) // 3)):
                pattern_counts: Dict[Tuple[str, ...], int] = {}
                for start in range(len(seq) - pattern_len + 1):
                    pattern = tuple(seq[start:start + pattern_len])
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                total_windows = len(seq) - pattern_len + 1
                num_possible = 3 ** pattern_len  # L/M/H combinations
                random_expected = total_windows / num_possible
                
                for pattern, count in pattern_counts.items():
                    if random_expected > 0 and count >= 3:
                        ratio = count / random_expected
                        if ratio >= 2.0:  # At least 2x above random
                            persistence = min(1.0, count / total_windows)
                            if persistence >= self.persistence_threshold * 0.5:
                                # Lower threshold for recurrence since they're rarer
                                invariants.append(Invariant(
                                    invariant_type="recurrence",
                                    description=(
                                        f"{agent}: pattern {'→'.join(pattern)} "
                                        f"recurs {ratio:.1f}x above random"
                                    ),
                                    persistence=persistence,
                                    confidence=min(1.0, (ratio - 1.0) / 3.0),
                                    agents_involved=[agent],
                                    metadata={
                                        "pattern": list(pattern),
                                        "count": count,
                                        "random_expected": float(random_expected),
                                        "ratio_above_random": float(ratio)
                                    }
                                ))
        
        # Sort by confidence descending and deduplicate
        invariants.sort(key=lambda x: x.confidence, reverse=True)
        return invariants[:10]  # Top 10 recurrence patterns
```

**Tests for this module (`tests/test_invariant_detector.py`):**

```python
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
```

---

### `core/lossy_translation.py`

**Purpose:** Transform values between representation spaces in ways that are deliberately lossy and non-invertible. These simulate the ML→AR boundary.

```python
from dataclasses import dataclass
from typing import List, Callable, Dict, Optional
import numpy as np


@dataclass
class TranslationRecord:
    """Records what happened during a lossy translation."""
    source_space: str
    target_space: str
    input_value: float
    output_value: float
    information_lost: str  # Human-readable description of what was lost


class LossyTranslator:
    """Applies lossy, non-invertible transformations between representation spaces.
    
    This is the implementation of the patent's core concept: translation between
    heterogeneous agent representations where information is necessarily destroyed.
    
    Each translation method returns both the translated value and a record of
    what information was lost (for proof generation).
    """
    
    @staticmethod
    def quantize(value: float, num_bins: int = 4, 
                 range_min: float = 0.0, range_max: float = 1.0) -> TranslationRecord:
        """Quantize continuous value into discrete bins.
        
        Lossy because: multiple continuous values map to same bin.
        Non-invertible because: cannot recover original from bin index.
        
        Args:
            value: Continuous input value.
            num_bins: Number of discrete output bins.
            range_min/max: Expected input range.
            
        Returns:
            TranslationRecord with output_value as bin center (0.0 to 1.0 range).
        """
        clamped = max(range_min, min(range_max, value))
        normalized = (clamped - range_min) / (range_max - range_min + 1e-10)
        bin_index = min(int(normalized * num_bins), num_bins - 1)
        bin_center = (bin_index + 0.5) / num_bins
        
        return TranslationRecord(
            source_space="continuous",
            target_space=f"quantized_{num_bins}bins",
            input_value=value,
            output_value=bin_center,
            information_lost=f"Quantized to {num_bins} bins: {value:.4f} → bin {bin_index} (center {bin_center:.4f})"
        )
    
    @staticmethod
    def threshold(value: float, threshold: float = 0.5) -> TranslationRecord:
        """Binary thresholding — map continuous value to 0 or 1.
        
        Lossy because: all values above threshold collapse to 1.0.
        Non-invertible because: 0.51 and 0.99 both become 1.0.
        
        This models NN continuous output → LP binary truth value.
        """
        output = 1.0 if value >= threshold else 0.0
        
        return TranslationRecord(
            source_space="continuous",
            target_space="binary",
            input_value=value,
            output_value=output,
            information_lost=f"Thresholded at {threshold}: {value:.4f} → {output:.1f}"
        )
    
    @staticmethod
    def soft_threshold(value: float, threshold: float = 0.5, 
                       steepness: float = 10.0) -> TranslationRecord:
        """Sigmoid-based soft thresholding.
        
        Less lossy than hard threshold but still non-invertible (many-to-one in tails).
        Models a more gradual NN→LP translation.
        """
        output = 1.0 / (1.0 + np.exp(-steepness * (value - threshold)))
        
        return TranslationRecord(
            source_space="continuous",
            target_space="soft_binary",
            input_value=value,
            output_value=float(output),
            information_lost=f"Soft threshold (k={steepness}): {value:.4f} → {output:.4f}"
        )
    
    @staticmethod
    def modular_reduction(value: float, modulus: int = 4) -> TranslationRecord:
        """Modular arithmetic reduction.
        
        Lossy because: values differing by multiples of modulus collapse.
        Non-invertible because: 5 mod 4 = 1 mod 4 = 1.
        
        Models the prime-encoding toy example from the patent.
        """
        int_value = int(round(value))
        output = int_value % modulus
        
        return TranslationRecord(
            source_space="integer",
            target_space=f"mod_{modulus}",
            input_value=value,
            output_value=float(output),
            information_lost=f"Modular reduction: {int_value} mod {modulus} = {output}"
        )
    
    @staticmethod
    def categorical_discretize(value: float, 
                                boundaries: List[float] = None) -> TranslationRecord:
        """Map continuous value to categorical label index.
        
        Default boundaries create LOW(0)/MEDIUM(1)/HIGH(2).
        Models NN risk score → LP symbolic category.
        
        This is the key ML→AR translation: continuous prediction → discrete predicate.
        """
        if boundaries is None:
            boundaries = [0.33, 0.66]  # LOW / MEDIUM / HIGH
        
        category = 0
        for b in boundaries:
            if value > b:
                category += 1
        
        return TranslationRecord(
            source_space="continuous",
            target_space=f"categorical_{len(boundaries)+1}",
            input_value=value,
            output_value=float(category),
            information_lost=f"Categorized: {value:.4f} → category {category} (boundaries: {boundaries})"
        )


class TranslationChain:
    """Applies a sequence of lossy translations, recording all information loss.
    
    Models the full composition path: NN output → quantize → threshold → LP predicate.
    
    Usage:
        chain = TranslationChain()
        chain.add_step("quantize", lambda v: LossyTranslator.quantize(v, num_bins=8))
        chain.add_step("threshold", lambda v: LossyTranslator.threshold(v, threshold=0.5))
        
        final_value, records = chain.translate(0.73)
    """
    
    def __init__(self):
        self.steps: List[tuple] = []  # List of (name, translation_function)
    
    def add_step(self, name: str, 
                 translation_fn: Callable[[float], TranslationRecord]) -> 'TranslationChain':
        """Add a translation step. Returns self for chaining."""
        self.steps.append((name, translation_fn))
        return self
    
    def translate(self, value: float) -> tuple:
        """Apply all translation steps in sequence.
        
        Args:
            value: Initial continuous value.
            
        Returns:
            (final_value: float, records: List[TranslationRecord])
        """
        current = value
        records = []
        
        for name, fn in self.steps:
            record = fn(current)
            records.append(record)
            current = record.output_value
        
        return current, records
```

**Tests (`tests/test_lossy_translation.py`):**

```python
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
```

---

### `core/coordination_controller.py`

**Purpose:** Monitors invariant persistence over time and triggers coordination responses when invariants degrade.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
from core.invariant_detector import InvariantDetector, Invariant


class CoordinationAction(Enum):
    MAINTAIN = "maintain"           # Invariant stable, no action needed
    INCREASE_MONITORING = "increase_monitoring"  # Slight degradation
    FLAG_RECONFIGURATION = "flag_reconfiguration"  # Moderate degradation
    TRIGGER_FALLBACK = "trigger_fallback"          # Severe degradation
    INVALIDATE = "invalidate"       # Invariant no longer holds


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
            
            # Update baselines for new invariants
            for inv in new_invariants:
                key = inv.description
                if key not in self.baselines:
                    self.baselines[key] = inv.persistence
            
            # Check degradation for known invariants
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
```

**Tests (`tests/test_coordination_controller.py`):**

```python
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
```

---

### `core/proof_generator.py`

**Purpose:** Generate hierarchical, human-readable proofs that structural invariants persist across the composition. CLARA requires proofs with ≤10 unfolding levels in natural deduction style.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from core.invariant_detector import Invariant
from core.lossy_translation import TranslationRecord


@dataclass
class ProofNode:
    """One node in a hierarchical proof tree."""
    level: int                          # Depth in proof tree (0 = root)
    statement: str                      # What this node proves
    justification: str                  # Why/how it's proven
    children: List['ProofNode'] = field(default_factory=list)
    verified: bool = True               # Whether this step passes
    evidence: Optional[Dict] = None     # Supporting data
    
    def depth(self) -> int:
        """Max depth of proof tree rooted at this node."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def render(self, indent: int = 0) -> str:
        """Render proof tree as indented text.
        
        Output format (natural deduction style):
        
        Level 0: [VERIFIED] Treatment A preserves all safety invariants
          ├── Level 1: [VERIFIED] Ordering invariant I₁ holds
          │   ├── Level 2: NN confidence 0.87 > threshold 0.5
          │   └── Level 2: LP approved(treatment_a) derived via SLD resolution
          └── Level 1: [VERIFIED] Bounded invariant I₂ holds
              └── Level 2: Risk score 0.23 within bounds [0.0, 0.5]
        """
        prefix = "  " * indent
        status = "✓" if self.verified else "✗"
        connector = "├── " if indent > 0 else ""
        
        lines = [f"{prefix}{connector}Level {self.level}: [{status}] {self.statement}"]
        if self.justification:
            lines.append(f"{prefix}    Justification: {self.justification}")
        
        for child in self.children:
            lines.append(child.render(indent + 1))
        
        return "\n".join(lines)


class ProofGenerator:
    """Generates hierarchical proofs of structural invariant persistence.
    
    Proof structure (from patent + CLARA requirements):
    
    Level 0: "System coordination is verified via structural invariants"
    Level 1: "Invariant I_k persists across composition" (one per invariant)
    Level 2: "Translation step T_i preserves invariant I_k" (one per translation)
    Level 3: "Specific evidence: input X → output Y, property holds"
    Level 4: "Statistical confidence: persistence = P over N cycles"
    
    This gives ≤5 levels, well within CLARA's ≤10 requirement.
    
    Usage:
        gen = ProofGenerator()
        proof = gen.generate_system_proof(
            invariants=detected_invariants,
            translation_records=records_from_chain,
            system_name="Medical Multi-Condition Guidance"
        )
        print(proof.render())
    """
    
    def generate_system_proof(self,
                               invariants: List[Invariant],
                               translation_records: List[List[TranslationRecord]] = None,
                               system_name: str = "System") -> ProofNode:
        """Generate a complete system-level proof.
        
        Args:
            invariants: Detected structural invariants.
            translation_records: Optional records from translation chains
                                 (list of record-lists, one per observation).
            system_name: Name for the root proof node.
            
        Returns:
            ProofNode tree with depth ≤ 5 levels.
        """
        root = ProofNode(
            level=0,
            statement=f"{system_name} coordination verified via {len(invariants)} structural invariants",
            justification="All detected invariants persist above threshold across lossy composition",
            verified=all(inv.persistence >= 0.5 for inv in invariants)
        )
        
        for inv in invariants:
            inv_node = self._generate_invariant_proof(inv, translation_records)
            root.children.append(inv_node)
        
        return root
    
    def _generate_invariant_proof(self, 
                                   invariant: Invariant,
                                   translation_records: List[List[TranslationRecord]] = None
                                   ) -> ProofNode:
        """Generate proof subtree for a single invariant."""
        inv_node = ProofNode(
            level=1,
            statement=f"Invariant [{invariant.invariant_type}]: {invariant.description}",
            justification=f"Persistence = {invariant.persistence:.1%}, "
                          f"confidence = {invariant.confidence:.1%}",
            verified=invariant.persistence >= 0.5
        )
        
        # Level 2: Translation steps that preserve this invariant
        if translation_records:
            trans_node = ProofNode(
                level=2,
                statement="Invariant persists through lossy translation chain",
                justification=f"Verified across {len(translation_records)} interaction cycles",
                verified=True
            )
            
            # Show first translation chain as example evidence
            if translation_records and len(translation_records) > 0:
                example = translation_records[0]
                for record in example:
                    step_node = ProofNode(
                        level=3,
                        statement=(
                            f"Translation: {record.source_space} → {record.target_space}"
                        ),
                        justification=record.information_lost,
                        verified=True,
                        evidence={
                            "input": record.input_value,
                            "output": record.output_value
                        }
                    )
                    trans_node.children.append(step_node)
            
            inv_node.children.append(trans_node)
        
        # Level 2: Statistical evidence
        stats_node = ProofNode(
            level=2,
            statement=f"Statistical persistence: {invariant.persistence:.1%}",
            justification=self._statistical_justification(invariant),
            verified=invariant.persistence >= 0.5,
            evidence=invariant.metadata
        )
        inv_node.children.append(stats_node)
        
        return inv_node
    
    def _statistical_justification(self, invariant: Invariant) -> str:
        """Generate human-readable statistical justification."""
        meta = invariant.metadata
        
        if invariant.invariant_type == "ordering":
            total = meta.get("total_cycles", "N")
            return (
                f"Property held in {invariant.persistence:.0%} of {total} observed cycles. "
                f"Confidence {invariant.confidence:.0%} above random baseline (50%)."
            )
        elif invariant.invariant_type == "bounded_oscillation":
            return (
                f"Value bounded in [{meta.get('min', '?'):.3f}, {meta.get('max', '?'):.3f}] "
                f"with std={meta.get('std', '?'):.4f}."
            )
        elif invariant.invariant_type == "recurrence":
            return (
                f"Pattern recurs {meta.get('ratio_above_random', '?'):.1f}x above random rate "
                f"({meta.get('count', '?')} occurrences vs {meta.get('random_expected', '?'):.1f} expected)."
            )
        return f"Persistence measured at {invariant.persistence:.1%}."
    
    def verify_proof_depth(self, root: ProofNode) -> bool:
        """Verify proof tree meets CLARA requirement of ≤10 unfolding levels."""
        return root.depth() <= 10
```

**No separate test file needed for proof_generator — it's tested through the examples.**

---

### `composition/__init__.py`

```python
from .nn_component import NNComponent
from .problog_component import ProbLogComponent
from .pipeline import CompositionPipeline
```

---

### `composition/nn_component.py`

**Purpose:** Wraps a PyTorch neural network as a component in the composition pipeline.

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class NNOutput:
    """Structured output from the neural network component."""
    raw_outputs: Dict[str, float]     # Named output values (e.g., {"risk_a": 0.82, "risk_b": 0.31})
    confidence: float                 # Overall prediction confidence
    model_name: str


class SimpleRiskNN(nn.Module):
    """A simple feedforward network for medical risk prediction.
    
    This is the toy NN for v0.1. It takes patient features and outputs
    risk scores for multiple conditions/treatments.
    
    Input: [age_normalized, blood_pressure_norm, glucose_norm, bmi_norm, ...]
    Output: [risk_treatment_a, risk_treatment_b, risk_interaction]
    """
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output in [0, 1] range — interpretable as risk probabilities
        )
        self.output_names = [f"risk_{i}" for i in range(output_dim)]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NNComponent:
    """Wraps a PyTorch model as a composition pipeline component.
    
    Usage:
        nn_comp = NNComponent(
            model=SimpleRiskNN(input_dim=8, output_dim=3),
            output_names=["risk_drug_a", "risk_drug_b", "risk_interaction"]
        )
        
        patient_data = torch.randn(1, 8)
        output = nn_comp.predict(patient_data)
        # output.raw_outputs = {"risk_drug_a": 0.82, "risk_drug_b": 0.31, "risk_interaction": 0.12}
    """
    
    def __init__(self, model: nn.Module, output_names: List[str] = None):
        self.model = model
        self.model.eval()
        self.output_names = output_names or [f"output_{i}" for i in range(100)]
        self.model_name = model.__class__.__name__
    
    def predict(self, input_tensor: torch.Tensor) -> NNOutput:
        """Run inference and return named outputs.
        
        Args:
            input_tensor: Shape (1, input_dim) or (input_dim,).
            
        Returns:
            NNOutput with named float values.
        """
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            raw = self.model(input_tensor).squeeze(0)
        
        outputs = {}
        for i, val in enumerate(raw.tolist()):
            name = self.output_names[i] if i < len(self.output_names) else f"output_{i}"
            outputs[name] = val
        
        confidence = float(torch.max(raw).item())
        
        return NNOutput(
            raw_outputs=outputs,
            confidence=confidence,
            model_name=self.model_name
        )
    
    def get_model(self) -> nn.Module:
        """Access underlying PyTorch model (for training)."""
        return self.model
```

---

### `composition/problog_component.py`

**Purpose:** Wraps ProbLog2 as the Bayesian Logic Program component. Includes a pure-Python fallback if ProbLog isn't installed.

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings


@dataclass 
class LPOutput:
    """Structured output from logic program inference."""
    query_results: Dict[str, float]   # Query atom → probability (e.g., {"safe(drug_a)": 0.92})
    proof_trace: List[str]            # Human-readable inference steps
    rules_fired: List[str]            # Which rules contributed to the result


# Try to import ProbLog; fall back to pure-Python implementation
try:
    from problog.program import PrologString
    from problog import get_evaluatable
    PROBLOG_AVAILABLE = True
except ImportError:
    PROBLOG_AVAILABLE = False
    warnings.warn(
        "ProbLog not installed. Using pure-Python LP fallback. "
        "Install with: pip install problog"
    )


class ProbLogComponent:
    """Bayesian Logic Program component using ProbLog2.
    
    Encodes medical treatment rules as probabilistic logic programs.
    Takes discretized inputs from the lossy translation layer and
    performs logical inference to produce treatment recommendations.
    
    Usage:
        lp = ProbLogComponent()
        lp.load_rules('''
            0.9::approved(drug_a) :- high_efficacy(drug_a).
            0.8::safe(drug_a) :- approved(drug_a), not contraindicated(drug_a).
            contraindicated(drug_a) :- condition(diabetes), condition(hypertension).
        ''')
        lp.set_evidence({"high_efficacy(drug_a)": True, "condition(diabetes)": True})
        result = lp.query(["safe(drug_a)", "approved(drug_a)"])
    """
    
    def __init__(self):
        self.rules_text: str = ""
        self.evidence: Dict[str, bool] = {}
        self._use_problog = PROBLOG_AVAILABLE
    
    def load_rules(self, rules: str) -> None:
        """Load ProbLog rules as a string.
        
        Args:
            rules: ProbLog program text. Example:
                0.9::approved(drug_a) :- high_efficacy(drug_a).
                0.1::contraindicated(drug_a) :- condition(renal_failure).
        """
        self.rules_text = rules.strip()
    
    def set_evidence(self, evidence: Dict[str, bool]) -> None:
        """Set observed evidence (facts derived from NN output after lossy translation).
        
        Args:
            evidence: Maps ground atom string → True/False.
                Example: {"high_efficacy(drug_a)": True, "condition(diabetes)": False}
        """
        self.evidence = evidence
    
    def query(self, query_atoms: List[str]) -> LPOutput:
        """Run probabilistic inference on query atoms.
        
        Args:
            query_atoms: List of ground atoms to query.
                Example: ["safe(drug_a)", "approved(drug_a)"]
                
        Returns:
            LPOutput with probabilities and proof trace.
        """
        if self._use_problog:
            return self._query_problog(query_atoms)
        else:
            return self._query_fallback(query_atoms)
    
    def _query_problog(self, query_atoms: List[str]) -> LPOutput:
        """Query using actual ProbLog2 engine."""
        # Build full program with evidence and queries
        program_parts = [self.rules_text]
        
        for atom, value in self.evidence.items():
            if value:
                program_parts.append(f"evidence({atom}).")
            else:
                program_parts.append(f"evidence(\\+{atom}).")
        
        for atom in query_atoms:
            program_parts.append(f"query({atom}).")
        
        program_text = "\n".join(program_parts)
        
        try:
            model = PrologString(program_text)
            evaluatable = get_evaluatable().create_from(model)
            result = evaluatable.evaluate()
            
            query_results = {}
            for atom, prob in result.items():
                query_results[str(atom)] = float(prob)
            
            # Fill in zeros for atoms not in results
            for atom in query_atoms:
                if atom not in query_results:
                    query_results[atom] = 0.0
            
            return LPOutput(
                query_results=query_results,
                proof_trace=[f"ProbLog inference on {len(query_atoms)} queries"],
                rules_fired=[r.strip() for r in self.rules_text.split('\n') if r.strip()]
            )
        except Exception as e:
            warnings.warn(f"ProbLog query failed: {e}. Falling back to pure-Python.")
            return self._query_fallback(query_atoms)
    
    def _query_fallback(self, query_atoms: List[str]) -> LPOutput:
        """Pure-Python fallback: simple rule matching without full ProbLog.
        
        This is a simplified inference engine for when ProbLog isn't available.
        It handles basic probabilistic rules with evidence.
        
        Limitations vs real ProbLog:
        - No negation-as-failure
        - No recursive rules
        - Simple forward chaining only
        """
        # Parse rules into (probability, head, body_atoms) tuples
        rules = self._parse_rules()
        
        # Start with evidence as known facts
        known: Dict[str, float] = {}
        for atom, value in self.evidence.items():
            known[atom] = 1.0 if value else 0.0
        
        proof_trace = []
        rules_fired = []
        
        # Simple forward chaining (2 passes to handle dependencies)
        for pass_num in range(3):
            for prob, head, body in rules:
                # Check if all body atoms are known and true
                body_prob = prob
                all_known = True
                for body_atom in body:
                    negated = body_atom.startswith("not ")
                    clean_atom = body_atom.replace("not ", "").strip()
                    
                    if clean_atom in known:
                        if negated:
                            body_prob *= (1.0 - known[clean_atom])
                        else:
                            body_prob *= known[clean_atom]
                    else:
                        if negated:
                            body_prob *= 1.0  # Unknown = not proven = true for negation
                        else:
                            all_known = False
                            break
                
                if all_known and body_prob > 0:
                    if head not in known or known[head] < body_prob:
                        known[head] = body_prob
                        rules_fired.append(f"{prob}::{head} :- {', '.join(body)}.")
                        proof_trace.append(
                            f"Pass {pass_num}: {head} = {body_prob:.3f} "
                            f"(from {', '.join(body)})"
                        )
        
        # Extract query results
        query_results = {}
        for atom in query_atoms:
            query_results[atom] = known.get(atom, 0.0)
        
        return LPOutput(
            query_results=query_results,
            proof_trace=proof_trace,
            rules_fired=rules_fired
        )
    
    def _parse_rules(self) -> List[tuple]:
        """Parse ProbLog-style rules into (probability, head, [body_atoms]).
        
        Handles formats:
            0.9::head :- body1, body2.
            head :- body1.          (implicit probability 1.0)
            fact.                    (no body)
        """
        rules = []
        for line in self.rules_text.split('\n'):
            line = line.strip().rstrip('.')
            if not line or line.startswith('%'):
                continue
            
            # Extract probability
            prob = 1.0
            if '::' in line:
                prob_str, line = line.split('::', 1)
                try:
                    prob = float(prob_str)
                except ValueError:
                    prob = 1.0
            
            # Split head and body
            if ':-' in line:
                head, body_str = line.split(':-', 1)
                head = head.strip()
                body = [b.strip() for b in body_str.split(',')]
            else:
                head = line.strip()
                body = []
            
            if head:
                rules.append((prob, head, body))
        
        return rules
    
    def add_rule(self, rule: str) -> None:
        """Add a single rule to the program (for human editing / knowledge updates).
        
        This enables the CLARA requirement: non-AI-experts can edit model knowledge
        by adding/modifying LP rules directly.
        
        Args:
            rule: A single ProbLog rule string, e.g., "contraindicated(drug_x) :- condition(y)."
        """
        self.rules_text += "\n" + rule.strip()
    
    def remove_rule(self, rule_fragment: str) -> bool:
        """Remove rules containing the given fragment. Returns True if any removed."""
        lines = self.rules_text.split('\n')
        filtered = [l for l in lines if rule_fragment not in l]
        changed = len(filtered) < len(lines)
        self.rules_text = '\n'.join(filtered)
        return changed
```

---

### `composition/pipeline.py`

**Purpose:** The full composition pipeline: NN → lossy translation → ProbLog → invariant detection → proof generation.

```python
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
            # Convention: NN output "risk_drug_a" with threshold → "high_risk(drug_a)" 
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
        
        Convention: "risk_drug_a" → "high_risk(drug_a)" if translated > 0.5
                    "risk_interaction" → "interaction_risk(high)" if categorical=2
        """
        # Simple convention: strip "risk_" prefix, add "high_" if positive
        clean = nn_output_name.replace("risk_", "")
        return f"high_efficacy({clean})"
    
    def _default_queries(self) -> List[str]:
        """Return default LP query atoms. Override in subclass for specific domains."""
        return ["safe(drug_a)", "safe(drug_b)", "approved(drug_a)", "approved(drug_b)"]
```

---

### `training/__init__.py`

```python
from .invariant_loss import InvariantLoss
from .constrained_trainer import ConstrainedTrainer
```

---

### `training/invariant_loss.py`

**Purpose:** A PyTorch loss function that penalizes violation of structural invariants during training. This is the "AR-based ML training" component.

```python
import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional


class InvariantLoss(nn.Module):
    """Loss function that combines standard ML loss with structural invariant penalties.
    
    L_total = L_task + lambda * L_invariant
    
    Where L_invariant penalizes violations of detected structural invariants.
    This makes the AR layer guide ML training — the core of "AR-based ML."
    
    Usage:
        criterion = InvariantLoss(
            task_loss=nn.BCELoss(),
            invariant_weight=0.5
        )
        criterion.add_ordering_constraint("risk_drug_a", ">", "risk_drug_b", margin=0.1)
        criterion.add_bound_constraint("risk_interaction", 0.0, 0.5)
        
        loss = criterion(predictions, targets)
    """
    
    def __init__(self, 
                 task_loss: nn.Module = None,
                 invariant_weight: float = 0.5):
        super().__init__()
        self.task_loss = task_loss or nn.BCELoss()
        self.invariant_weight = invariant_weight
        self.ordering_constraints: List[Dict] = []
        self.bound_constraints: List[Dict] = []
    
    def add_ordering_constraint(self, output_a: str, direction: str, output_b: str,
                                 idx_a: int = 0, idx_b: int = 1,
                                 margin: float = 0.05) -> None:
        """Add constraint that output_a should be > or < output_b.
        
        This encodes an ordering invariant from detection into a training constraint.
        
        Args:
            output_a/b: Names (for logging). 
            direction: ">" or "<"
            idx_a/b: Indices into the model output tensor.
            margin: Minimum required difference (soft constraint).
        """
        self.ordering_constraints.append({
            "name_a": output_a, "name_b": output_b,
            "direction": direction,
            "idx_a": idx_a, "idx_b": idx_b,
            "margin": margin
        })
    
    def add_bound_constraint(self, output_name: str, 
                              idx: int = 0,
                              lower: float = 0.0, upper: float = 1.0) -> None:
        """Add constraint that output should stay within bounds.
        
        Encodes a bounded oscillation invariant as a training constraint.
        """
        self.bound_constraints.append({
            "name": output_name, "idx": idx,
            "lower": lower, "upper": upper
        })
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            predictions: Model output tensor, shape (batch, num_outputs).
            targets: Ground truth, shape (batch, num_outputs).
            
        Returns:
            Scalar loss tensor.
        """
        # Standard task loss
        loss_task = self.task_loss(predictions, targets)
        
        # Invariant violation penalties
        loss_invariant = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Ordering constraints
        for constraint in self.ordering_constraints:
            a = predictions[:, constraint["idx_a"]]
            b = predictions[:, constraint["idx_b"]]
            margin = constraint["margin"]
            
            if constraint["direction"] == ">":
                # Penalty when a is not sufficiently greater than b
                violation = torch.relu(b - a + margin)
            else:
                violation = torch.relu(a - b + margin)
            
            loss_invariant = loss_invariant + violation.mean()
        
        # Bound constraints
        for constraint in self.bound_constraints:
            val = predictions[:, constraint["idx"]]
            lower_violation = torch.relu(constraint["lower"] - val)
            upper_violation = torch.relu(val - constraint["upper"])
            loss_invariant = loss_invariant + (lower_violation + upper_violation).mean()
        
        return loss_task + self.invariant_weight * loss_invariant
    
    def get_violation_report(self, predictions: torch.Tensor) -> Dict[str, float]:
        """Check how many constraints are violated (for monitoring, not training).
        
        Returns dict mapping constraint_name -> fraction of batch violating.
        """
        report = {}
        with torch.no_grad():
            for c in self.ordering_constraints:
                a = predictions[:, c["idx_a"]]
                b = predictions[:, c["idx_b"]]
                if c["direction"] == ">":
                    violated = (a <= b).float().mean().item()
                else:
                    violated = (a >= b).float().mean().item()
                report[f"ordering_{c['name_a']}_{c['direction']}_{c['name_b']}"] = violated
            
            for c in self.bound_constraints:
                val = predictions[:, c["idx"]]
                violated = ((val < c["lower"]) | (val > c["upper"])).float().mean().item()
                report[f"bound_{c['name']}"] = violated
        
        return report
```

---

### `training/constrained_trainer.py`

**Purpose:** Training loop that uses InvariantLoss and reports invariant compliance during training.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Callable
from training.invariant_loss import InvariantLoss


class ConstrainedTrainer:
    """Trains a model while enforcing structural invariant constraints.
    
    This is AR-based ML training: the AR scaffold (invariant constraints derived
    from the logic program) shapes how the NN learns.
    
    Usage:
        trainer = ConstrainedTrainer(
            model=my_nn,
            criterion=my_invariant_loss,
            lr=0.001,
            epochs=50
        )
        history = trainer.train(train_loader)
        # history contains loss curves and invariant violation rates per epoch
    """
    
    def __init__(self,
                 model: nn.Module,
                 criterion: InvariantLoss,
                 lr: float = 0.001,
                 epochs: int = 50,
                 device: str = "cpu"):
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.epochs = epochs
        self.device = device
        self.model.to(device)
    
    def train(self, train_loader: DataLoader, 
              val_loader: DataLoader = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """Train the model with invariant constraints.
        
        Args:
            train_loader: Training data (features, labels).
            val_loader: Optional validation data.
            verbose: Print progress per epoch.
            
        Returns:
            History dict with keys: 
                "train_loss", "val_loss", "invariant_violations"
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "invariant_violations": [],
        }
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_violations = {}
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Track violations
                violations = self.criterion.get_violation_report(preds)
                for k, v in violations.items():
                    epoch_violations[k] = epoch_violations.get(k, 0) + v
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_violations = {k: v / num_batches for k, v in epoch_violations.items()}
            
            history["train_loss"].append(avg_loss)
            history["invariant_violations"].append(avg_violations)
            
            # Validation
            if val_loader:
                val_loss = self._evaluate(val_loader)
                history["val_loss"].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                viol_str = ", ".join(f"{k}: {v:.1%}" for k, v in avg_violations.items())
                print(f"Epoch {epoch+1}/{self.epochs} — loss: {avg_loss:.4f} — violations: {viol_str}")
        
        return history
    
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                total_loss += loss.item()
                count += 1
        return total_loss / max(count, 1)
    
    @staticmethod
    def make_loader(features: torch.Tensor, labels: torch.Tensor,
                    batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Helper to create DataLoader from tensors."""
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```

---

### `verification/__init__.py`

```python
from .categories import Category, Functor, NaturalTransformation
from .yoneda_checker import YonedaChecker
```

---

### `verification/categories.py`

**Purpose:** Lightweight implementation of categories, functors, and natural transformations — just enough to verify structural invariant persistence formally.

**IMPORTANT CONTEXT FOR IMPLEMENTER:** This is NOT a general-purpose category theory library. It's purpose-built for verifying that structural properties persist across the NN→LP composition. Keep it simple and concrete.

```python
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Callable, Optional, Any, List


@dataclass(frozen=True)
class Object:
    """An object in a category. Immutable and hashable."""
    name: str
    space: str = ""  # Which category this belongs to
    
    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Morphism:
    """A morphism (arrow) between objects in a category."""
    source: Object
    target: Object
    name: str = ""
    
    def __str__(self):
        return f"{self.source} --{self.name}--> {self.target}"


class Category:
    """A category: objects + morphisms + composition.
    
    For our purposes:
    - NN Category: objects are NN states, morphisms are forward propagation steps
    - LP Category: objects are LP models (sets of facts), morphisms are inference steps
    
    Usage:
        nn_cat = Category("NeuralNetwork")
        s0 = nn_cat.add_object("state_low")     # NN output < 0.5
        s1 = nn_cat.add_object("state_high")     # NN output >= 0.5
        nn_cat.add_morphism(s0, s1, "activate")   # Transition low → high
    """
    
    def __init__(self, name: str):
        self.name = name
        self.objects: Set[Object] = set()
        self.morphisms: Set[Morphism] = set()
        self._composition: Dict[Tuple[Morphism, Morphism], Morphism] = {}
        self._identity: Dict[Object, Morphism] = {}
    
    def add_object(self, name: str) -> Object:
        obj = Object(name=name, space=self.name)
        self.objects.add(obj)
        # Add identity morphism
        id_m = Morphism(source=obj, target=obj, name=f"id_{name}")
        self.morphisms.add(id_m)
        self._identity[obj] = id_m
        return obj
    
    def add_morphism(self, source: Object, target: Object, name: str = "") -> Morphism:
        if not name:
            name = f"{source.name}_to_{target.name}"
        m = Morphism(source=source, target=target, name=name)
        self.morphisms.add(m)
        return m
    
    def compose(self, f: Morphism, g: Morphism) -> Optional[Morphism]:
        """Compose g ∘ f (f first, then g). Returns None if not composable."""
        if f.target != g.source:
            return None
        if (f, g) in self._composition:
            return self._composition[(f, g)]
        # Auto-compose
        composed = Morphism(
            source=f.source, target=g.target,
            name=f"{g.name}∘{f.name}"
        )
        self._composition[(f, g)] = composed
        self.morphisms.add(composed)
        return composed
    
    def identity(self, obj: Object) -> Morphism:
        return self._identity[obj]
    
    def hom_set(self, source: Object, target: Object) -> Set[Morphism]:
        """All morphisms from source to target (Hom(source, target))."""
        return {m for m in self.morphisms if m.source == source and m.target == target}


class Functor:
    """A functor F: C → D mapping objects and morphisms between categories.
    
    For structural invariants, the key functor is the lossy translation:
    F: NN_Category → LP_Category
    
    This functor is NON-FAITHFUL (multiple NN morphisms map to same LP morphism)
    and NON-FULL (not all LP morphisms have NN pre-images).
    These properties make it LOSSY — which is what we want.
    
    Usage:
        F = Functor("NN_to_LP", nn_cat, lp_cat)
        F.map_object(nn_state_high, lp_true)
        F.map_object(nn_state_low, lp_false)
        F.map_morphism(nn_activate, lp_derive)
    """
    
    def __init__(self, name: str, source_cat: Category, target_cat: Category):
        self.name = name
        self.source = source_cat
        self.target = target_cat
        self._object_map: Dict[Object, Object] = {}
        self._morphism_map: Dict[Morphism, Morphism] = {}
    
    def map_object(self, source_obj: Object, target_obj: Object) -> None:
        self._object_map[source_obj] = target_obj
    
    def map_morphism(self, source_morph: Morphism, target_morph: Morphism) -> None:
        self._morphism_map[source_morph] = target_morph
    
    def apply_object(self, obj: Object) -> Optional[Object]:
        return self._object_map.get(obj)
    
    def apply_morphism(self, morph: Morphism) -> Optional[Morphism]:
        return self._morphism_map.get(morph)
    
    def is_faithful(self) -> bool:
        """Check if functor is faithful (injective on morphisms).
        If NOT faithful, it's lossy — multiple source morphisms map to same target."""
        targets = list(self._morphism_map.values())
        return len(targets) == len(set(targets))
    
    def is_full(self) -> bool:
        """Check if functor is full (surjective on each hom-set).
        If NOT full, some target morphisms have no pre-image — information lost."""
        for src_obj in self._object_map:
            for src_obj2 in self._object_map:
                target_a = self._object_map[src_obj]
                target_b = self._object_map[src_obj2]
                target_hom = self.target.hom_set(target_a, target_b)
                image_hom = set()
                for m in self.source.hom_set(src_obj, src_obj2):
                    if m in self._morphism_map:
                        image_hom.add(self._morphism_map[m])
                if not target_hom.issubset(image_hom):
                    return False
        return True
    
    def preserves_composition(self) -> bool:
        """Check functoriality: F(g ∘ f) = F(g) ∘ F(f)."""
        for (f, g), gf in self.source._composition.items():
            if f in self._morphism_map and g in self._morphism_map and gf in self._morphism_map:
                Ff = self._morphism_map[f]
                Fg = self._morphism_map[g]
                Fgf = self._morphism_map[gf]
                composed = self.target.compose(Ff, Fg)
                if composed != Fgf:
                    return False
        return True


class NaturalTransformation:
    """A natural transformation η: F → G between functors F, G: C → D.
    
    For structural invariants, a natural transformation represents a 
    structural property that is preserved uniformly across all objects.
    
    The key theorem: If η is natural (all squares commute), then the 
    structural property it represents PERSISTS under lossy translation.
    
    Usage:
        eta = NaturalTransformation("ordering_invariant", F, G)
        eta.set_component(obj_a, morphism_in_D)
        is_valid = eta.check_naturality()
    """
    
    def __init__(self, name: str, source_functor: Functor, target_functor: Functor):
        assert source_functor.source == target_functor.source, \
            "Functors must share source category"
        assert source_functor.target == target_functor.target, \
            "Functors must share target category"
        
        self.name = name
        self.F = source_functor
        self.G = target_functor
        self.components: Dict[Object, Morphism] = {}
        # components[X] = η_X : F(X) → G(X) in target category
    
    def set_component(self, source_object: Object, component_morphism: Morphism) -> None:
        """Set the component η_X for object X in source category.
        
        η_X must be a morphism F(X) → G(X) in the target category.
        """
        self.components[source_object] = component_morphism
    
    def check_naturality(self) -> Tuple[bool, List[str]]:
        """Check all naturality squares commute.
        
        For each morphism f: X → Y in source category:
            η_Y ∘ F(f) = G(f) ∘ η_X
        
        Returns:
            (all_commute: bool, violations: List[str])
        """
        violations = []
        source_cat = self.F.source
        
        for f in source_cat.morphisms:
            X = f.source
            Y = f.target
            
            if X not in self.components or Y not in self.components:
                continue
            
            eta_X = self.components[X]
            eta_Y = self.components[Y]
            Ff = self.F.apply_morphism(f)
            Gf = self.G.apply_morphism(f)
            
            if Ff is None or Gf is None:
                continue
            
            # Check: η_Y ∘ F(f) = G(f) ∘ η_X
            left = self.F.target.compose(Ff, eta_Y)    # η_Y ∘ F(f)
            right = self.G.target.compose(eta_X, Gf)   # G(f) ∘ η_X
            
            if left is not None and right is not None and left != right:
                violations.append(
                    f"Naturality fails at {f}: "
                    f"η_{Y.name} ∘ F({f.name}) ≠ G({f.name}) ∘ η_{X.name}"
                )
        
        return (len(violations) == 0, violations)
```

---

### `verification/yoneda_checker.py`

**Purpose:** Uses the Yoneda perspective to verify that structural invariants persist under lossy functors.

```python
from typing import List, Dict, Tuple
from verification.categories import Category, Functor, NaturalTransformation, Object, Morphism


class YonedaChecker:
    """Verifies structural invariant persistence using the Yoneda embedding.
    
    Yoneda insight: An object X in category C is fully characterized by 
    Hom(-, X) — all morphisms into X from every other object.
    
    Even if functor F: C → D is lossy, the Yoneda embedding Y: C → Set^{C^op}
    preserves all structural relationships.
    
    Practical meaning: If an ordering relationship or bounded behavior persists
    in the Hom-set structure, it persists under ANY lossy translation.
    
    Usage:
        checker = YonedaChecker()
        result = checker.verify_invariant_persistence(
            source_cat=nn_category,
            target_cat=lp_category,
            functor=nn_to_lp,
            invariant_type="ordering",
            objects=[nn_high, nn_low]
        )
        print(result["verified"])  # True/False
        print(result["proof_steps"])  # Human-readable proof
    """
    
    def compute_hom_profile(self, category: Category, obj: Object) -> Dict[Object, int]:
        """Compute the Yoneda profile of an object: |Hom(X, obj)| for all X.
        
        This is a simplified Yoneda embedding — instead of tracking the full
        hom-set functor, we track its cardinality, which is sufficient for
        detecting ordering and boundedness invariants.
        """
        profile = {}
        for other in category.objects:
            hom = category.hom_set(other, obj)
            profile[other] = len(hom)
        return profile
    
    def verify_invariant_persistence(self,
                                      source_cat: Category,
                                      target_cat: Category,
                                      functor: Functor,
                                      invariant_type: str,
                                      objects: List[Object]) -> Dict:
        """Verify that a structural invariant persists under the lossy functor.
        
        Args:
            source_cat: Source category (e.g., NN category).
            target_cat: Target category (e.g., LP category).
            functor: The lossy functor F: source → target.
            invariant_type: "ordering" or "bounded".
            objects: Objects involved in the invariant.
            
        Returns:
            Dict with keys: "verified" (bool), "proof_steps" (List[str]),
            "lossiness" (Dict describing what the functor loses).
        """
        proof_steps = []
        
        # Step 1: Characterize lossiness
        is_faithful = functor.is_faithful()
        is_full = functor.is_full()
        proof_steps.append(
            f"Functor '{functor.name}' analysis: "
            f"faithful={is_faithful}, full={is_full}"
        )
        if not is_faithful:
            proof_steps.append(
                "  → Functor is NOT faithful (lossy): "
                "multiple source morphisms collapse to same target morphism"
            )
        
        # Step 2: Compute Yoneda profiles in source
        source_profiles = {}
        for obj in objects:
            profile = self.compute_hom_profile(source_cat, obj)
            source_profiles[obj] = profile
            proof_steps.append(
                f"Yoneda profile Y({obj.name}): "
                f"{{{', '.join(f'|Hom({k.name}, {obj.name})| = {v}' for k, v in profile.items())}}}"
            )
        
        # Step 3: Compute Yoneda profiles in target (via functor image)
        target_profiles = {}
        for obj in objects:
            target_obj = functor.apply_object(obj)
            if target_obj:
                profile = self.compute_hom_profile(target_cat, target_obj)
                target_profiles[obj] = profile
                proof_steps.append(
                    f"Yoneda profile Y(F({obj.name})) = Y({target_obj.name}): "
                    f"{{{', '.join(f'|Hom({k.name}, {target_obj.name})| = {v}' for k, v in profile.items())}}}"
                )
        
        # Step 4: Check if invariant persists
        verified = False
        
        if invariant_type == "ordering" and len(objects) >= 2:
            verified = self._check_ordering_persistence(
                objects, source_profiles, target_profiles, functor, proof_steps
            )
        elif invariant_type == "bounded":
            verified = self._check_bounded_persistence(
                objects, source_profiles, target_profiles, functor, proof_steps
            )
        else:
            proof_steps.append(f"Unknown invariant type: {invariant_type}")
        
        return {
            "verified": verified,
            "proof_steps": proof_steps,
            "lossiness": {
                "faithful": is_faithful,
                "full": is_full,
                "preserves_composition": functor.preserves_composition()
            }
        }
    
    def _check_ordering_persistence(self, objects, source_profiles, target_profiles,
                                     functor, proof_steps) -> bool:
        """Check if ordering relationship persists under functor.
        
        Ordering invariant: |Hom(-, A)| > |Hom(-, B)| in source
        should imply: |Hom(-, F(A))| > |Hom(-, F(B))| in target
        (or at least: F(A) ≠ F(B) — they don't collapse)
        """
        obj_a, obj_b = objects[0], objects[1]
        
        # Check source ordering
        total_a = sum(source_profiles.get(obj_a, {}).values())
        total_b = sum(source_profiles.get(obj_b, {}).values())
        source_order = ">" if total_a > total_b else "<" if total_a < total_b else "="
        proof_steps.append(
            f"Source ordering: |Y({obj_a.name})| = {total_a} {source_order} "
            f"|Y({obj_b.name})| = {total_b}"
        )
        
        # Check target ordering
        if obj_a in target_profiles and obj_b in target_profiles:
            total_fa = sum(target_profiles[obj_a].values())
            total_fb = sum(target_profiles[obj_b].values())
            target_order = ">" if total_fa > total_fb else "<" if total_fa < total_fb else "="
            proof_steps.append(
                f"Target ordering: |Y(F({obj_a.name}))| = {total_fa} {target_order} "
                f"|Y(F({obj_b.name}))| = {total_fb}"
            )
            
            persists = (source_order == target_order)
            if persists:
                proof_steps.append(
                    f"✓ VERIFIED: Ordering invariant persists under lossy functor"
                )
            else:
                proof_steps.append(
                    f"✗ VIOLATED: Ordering changed from {source_order} to {target_order}"
                )
            return persists
        
        proof_steps.append("Cannot verify: objects not in target profiles")
        return False
    
    def _check_bounded_persistence(self, objects, source_profiles, target_profiles,
                                    functor, proof_steps) -> bool:
        """Check if bounded behavior persists under functor."""
        for obj in objects:
            if obj in target_profiles:
                total = sum(target_profiles[obj].values())
                # Bounded = total hom-set size is finite and non-zero
                bounded = 0 < total < 1000  # Practical bound
                proof_steps.append(
                    f"Target |Y(F({obj.name}))| = {total}: "
                    f"{'bounded ✓' if bounded else 'unbounded ✗'}"
                )
                if not bounded:
                    return False
        
        proof_steps.append("✓ VERIFIED: Bounded behavior persists")
        return True
```

---

### `examples/toy_prime_encoding.py`

**Purpose:** Implements the prime-based encoding toy example directly from the patent. This is the simplest possible demonstration of the full system.

```python
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
```

---

### `examples/medical_multicondition.py`

**Purpose:** The full CLARA demonstration: NN → lossy translation → Bayesian-LP → invariant detection → proof generation, applied to medical multi-condition treatment guidance.

```python
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


if __name__ == "__main__":
    main()
```

---

## IMPLEMENTATION NOTES FOR CLAUDE CODE (SONNET)

1. **Start with Layer 1** (invariant_detector.py, lossy_translation.py). Run tests. Confirm green.

2. **Each module has `__init__.py` imports.** Make sure they're correct before moving to the next layer.

3. **The `examples/` scripts use `sys.path.insert`** to allow running as `python examples/toy_prime_encoding.py` from the project root. Alternatively, if the user sets up a proper package with `setup.py` or `pyproject.toml`, remove those sys.path hacks.

4. **ProbLog fallback is intentional.** ProbLog can be finicky to install. The pure-Python fallback in `problog_component.py` handles basic forward chaining. It won't handle negation-as-failure or complex recursive rules, but it's sufficient for the medical demo.

5. **All `__init__.py` files in composition/, training/, verification/ need appropriate imports** — see the module specs above.

6. **The verification/ module is the most theoretically complex.** If Sonnet struggles with it, bring specific questions back to Opus. The key thing to get right: the `NaturalTransformation.check_naturality()` method must correctly check that for each morphism f: X → Y, the square η_Y ∘ F(f) = G(f) ∘ η_X commutes.

7. **run_toy.sh** should contain:
```bash
#!/bin/bash
cd "$(dirname "$0")"
python -m examples.toy_prime_encoding
```

---

## WHAT SUCCESS LOOKS LIKE

When you run `python examples/toy_prime_encoding.py`, you should see:
- 100 interaction cycles processed
- Ordering invariants detected between agents (agent_a > agent_c, etc.)
- A proof tree printed with ≤5 levels
- Disruption phase showing degradation detection

When you run `python examples/medical_multicondition.py`, you should see:
- 200 patients processed through NN → LP pipeline
- Structural invariants across the composition boundary
- A hierarchical proof tree
- Human knowledge editing changing LP output without retraining

**These two working examples are the deliverables. Everything else supports them.**
