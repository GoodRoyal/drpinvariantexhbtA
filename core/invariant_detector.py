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
