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
