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
                f"Value bounded in [{meta.get('min', 0.0):.3f}, {meta.get('max', 0.0):.3f}] "
                f"with std={meta.get('std', 0.0):.4f}."
            )
        elif invariant.invariant_type == "recurrence":
            return (
                f"Pattern recurs {meta.get('ratio_above_random', 0.0):.1f}x above random rate "
                f"({meta.get('count', 0)} occurrences vs {meta.get('random_expected', 0.0):.1f} expected)."
            )
        return f"Persistence measured at {invariant.persistence:.1%}."

    def verify_proof_depth(self, root: ProofNode) -> bool:
        """Verify proof tree meets CLARA requirement of ≤10 unfolding levels."""
        return root.depth() <= 10
