# Structural Invariants Under Lossy Translation

Reference implementation of **"Coordination of Heterogeneous Agents by Discovering Structural Invariants Under Lossy Translation"** (Paredes), targeting [DARPA CLARA TA1](https://www.darpa.mil/program/contract-based-learning-architectures-and-reasoning-for-autonomy).

`EXHIBIT_A.md` contains unedited live terminal output from both integration examples — invariant detection, hierarchical proof trees, degradation response, and human knowledge editing.

## What this demonstrates

Heterogeneous AI systems — mixing neural networks (ML) with logic programs (AR) — communicate across a boundary where translation is **lossy and non-invertible**. Despite information loss at that boundary, certain structural properties of the agent interaction persist. This framework discovers those properties automatically, verifies them formally, generates CLARA-compliant natural-deduction proof trees (≤ 10 unfolding levels), and signals when they degrade.

### CLARA TA1 capabilities shown

| Capability | Status |
|---|---|
| ML + AR composition (Neural Network + Bayesian-LP) | Implemented |
| Lossy translation at composition boundary | Implemented (threshold, quantize, modular reduction, categorical, soft-threshold) |
| Automatic invariant detection (ordering, bounded oscillation, recurrence) | Implemented |
| Hierarchical proof generation (natural deduction, ≤ 10 levels) | Implemented |
| Invariant degradation detection + coordinated response | Implemented |
| Human-editable AR knowledge without ML retraining | Implemented |
| AR-guided ML training (InvariantLoss) | Implemented |
| Category-theoretic verification (Yoneda Lemma) | Implemented |

## Architecture

```
structural-invariants/
├── core/                        # Layer 1 + 2 — invariant engine
│   ├── invariant_detector.py    # Ordering, bounded-oscillation, recurrence detection
│   ├── lossy_translation.py     # Quantize, threshold, modular reduction, categorical
│   ├── coordination_controller.py  # Degradation thresholds → MAINTAIN / INCREASE_MONITORING
│   │                                               # / FLAG_RECONFIGURATION / TRIGGER_FALLBACK
│   └── proof_generator.py       # Natural-deduction proof trees (ProofNode, ProofGenerator)
│
├── composition/                 # Layer 3 + 4 — NN × LP pipeline
│   ├── nn_component.py          # SimpleRiskNN (PyTorch), NNComponent wrapper
│   ├── problog_component.py     # ProbLogComponent (ProbLog2 + pure-Python fallback)
│   └── pipeline.py              # CompositionPipeline — NN → translation → LP → controller
│
├── training/                    # Layer 4 — AR-guided ML training
│   ├── invariant_loss.py        # InvariantLoss(nn.Module) — ordering + bound hinge penalties
│   └── constrained_trainer.py  # ConstrainedTrainer with Adam, train/val history
│
├── verification/                # Layer 5 — category-theoretic verification
│   ├── categories.py            # Object, Morphism, Category, Functor, NaturalTransformation
│   └── yoneda_checker.py        # YonedaChecker — Hom-profile ordering + boundedness
│
├── examples/                    # Layer 6 — integration demonstrations
│   ├── toy_prime_encoding.py    # Prime-encoding toy scenario (patent example)
│   └── medical_multicondition.py  # CLARA TA1 full demo — 200 patients
│
├── tests/                       # 91 tests, all passing
├── EXHIBIT_A.md                 # Full unedited terminal output (proposal exhibit)
└── README.md                    # This file
```

## Quickstart

```bash
# Install dependencies (requires uv)
uv sync

# Run the patent toy example
uv run python examples/toy_prime_encoding.py

# Run the CLARA TA1 medical demonstration
uv run python examples/medical_multicondition.py

# Run the full test suite
uv run pytest tests/ -q
```

Expected test output:
```
91 passed, 23 warnings in 8.61s
```

The warnings are informational: ProbLog2 rejects queries when evidence atoms have no defined clauses and the system falls back to pure-Python forward chaining. All tests pass.

## Key concepts

### Structural invariants

Three classes of invariant are detected from a sliding observation window:

- **Ordering**: `agent_a > agent_b` holds in X% of cycles — persists through lossy translation because relative orderings survive threshold and quantization operators
- **Bounded oscillation**: a signal stays within `[min, max]` with low std/mean ratio — survives translation when bounds bracket the translation threshold
- **Recurrence**: a discretized symbol sequence (L/M/H) repeats far above chance — survives because the underlying generative pattern dominates over translation noise

### Lossy translation

Five translation operators model the ML→AR boundary:

| Operator | Information destroyed |
|---|---|
| `threshold(x, t)` | Magnitude — only sign relative to t survives |
| `quantize(x, n)` | Fine gradations — n-bin discretization |
| `modular_reduction(x, m)` | Quotient — only remainder survives |
| `categorical_discretize(x, boundaries)` | Within-band variation |
| `soft_threshold(x, t, s)` | Smooth version of threshold |

### Coordination actions

Degradation is mapped to four action levels:

| Degradation | Action |
|---|---|
| < 10% | `MAINTAIN` |
| 10–20% | `INCREASE_MONITORING` |
| 20–30% | `FLAG_RECONFIGURATION` |
| > 30% | `TRIGGER_FALLBACK` |

### CLARA proof depth

All proofs generated in the examples are ≤ 4 levels deep, well within the CLARA requirement of ≤ 10 unfolding levels.

## Dependencies

- Python ≥ 3.10
- PyTorch
- NumPy
- ProbLog2 (optional — pure-Python fallback is used automatically when unavailable or when evidence atoms have no defined clauses)
- pytest (for tests)

## License

Apache License 2.0. See [LICENSE](LICENSE).
