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
