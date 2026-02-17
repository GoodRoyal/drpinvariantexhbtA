import pytest
from verification.categories import Category, Functor, Object
from verification.yoneda_checker import YonedaChecker


# ── shared fixture builders ───────────────────────────────────────────────────

def build_nn_lp_scenario():
    """
    Build a minimal NN→LP categorical scenario.

    NN category:  state_high --activate--> state_active
                  state_low  (no outgoing morphisms to active)
    LP category:  lp_true  --derive--> lp_active
                  lp_false

    Functor: state_high → lp_true, state_low → lp_false
             activate → derive
    The functor is faithful (one morphism maps to one morphism).
    state_high has more incoming morphisms than state_low, preserving ordering.
    """
    nn = Category("NN")
    sh = nn.add_object("state_high")
    sl = nn.add_object("state_low")
    sa = nn.add_object("state_active")
    act = nn.add_morphism(sh, sa, "activate")
    # extra morphism into state_high (gives it a higher hom-set total)
    nn.add_morphism(sl, sh, "upgrade")

    lp = Category("LP")
    lt = lp.add_object("lp_true")
    lf = lp.add_object("lp_false")
    la = lp.add_object("lp_active")
    derive = lp.add_morphism(lt, la, "derive")
    # mirror the extra morphism in LP
    lp.add_morphism(lf, lt, "assert")

    F = Functor("NN_to_LP", nn, lp)
    F.map_object(sh, lt)
    F.map_object(sl, lf)
    F.map_object(sa, la)
    F.map_morphism(act, derive)

    return nn, lp, F, sh, sl, sa, lt, lf


# ── compute_hom_profile ───────────────────────────────────────────────────────

def test_hom_profile_counts_identity():
    """Every object has at least its own identity morphism in its hom-profile."""
    cat = Category("C")
    a = cat.add_object("A")
    checker = YonedaChecker()
    profile = checker.compute_hom_profile(cat, a)
    # Hom(A, A) includes id_A
    assert profile[a] >= 1


def test_hom_profile_reflects_extra_morphisms():
    """An object with more incoming morphisms should have a larger profile total."""
    cat = Category("C")
    a = cat.add_object("A")
    b = cat.add_object("B")
    # Two morphisms into b, only identity into a
    cat.add_morphism(a, b, "f1")
    cat.add_morphism(a, b, "f2")
    checker = YonedaChecker()
    profile_a = checker.compute_hom_profile(cat, a)
    profile_b = checker.compute_hom_profile(cat, b)
    assert sum(profile_b.values()) > sum(profile_a.values())


# ── verify_invariant_persistence — ordering ───────────────────────────────────

def test_ordering_verified_when_preserved():
    """When source ordering persists in target, verified=True."""
    nn, lp, F, sh, sl, sa, lt, lf = build_nn_lp_scenario()
    checker = YonedaChecker()
    result = checker.verify_invariant_persistence(
        source_cat=nn,
        target_cat=lp,
        functor=F,
        invariant_type="ordering",
        objects=[sh, sl]
    )
    assert result["verified"] is True
    assert any("VERIFIED" in step for step in result["proof_steps"])


def test_ordering_result_has_required_keys():
    """Result dict must have verified, proof_steps, lossiness."""
    nn, lp, F, sh, sl, *_ = build_nn_lp_scenario()
    checker = YonedaChecker()
    result = checker.verify_invariant_persistence(
        source_cat=nn, target_cat=lp, functor=F,
        invariant_type="ordering", objects=[sh, sl]
    )
    assert "verified" in result
    assert "proof_steps" in result
    assert "lossiness" in result


def test_lossiness_report_has_expected_keys():
    nn, lp, F, sh, sl, *_ = build_nn_lp_scenario()
    checker = YonedaChecker()
    result = checker.verify_invariant_persistence(
        source_cat=nn, target_cat=lp, functor=F,
        invariant_type="ordering", objects=[sh, sl]
    )
    lossiness = result["lossiness"]
    assert "faithful" in lossiness
    assert "full" in lossiness
    assert "preserves_composition" in lossiness


def test_ordering_not_verified_when_reversed():
    """When the functor maps both objects to the same target, ordering can't persist."""
    nn = Category("NN")
    sh = nn.add_object("state_high")
    sl = nn.add_object("state_low")
    nn.add_morphism(sl, sh, "upgrade")  # sh has more incoming

    lp = Category("LP")
    lt = lp.add_object("lp_true")

    # Collapse: both objects map to the same target object
    F = Functor("collapse", nn, lp)
    F.map_object(sh, lt)
    F.map_object(sl, lt)

    checker = YonedaChecker()
    result = checker.verify_invariant_persistence(
        source_cat=nn, target_cat=lp, functor=F,
        invariant_type="ordering", objects=[sh, sl]
    )
    # Both map to same object so target profiles are identical → order = "="
    # Source has sh > sl, target has sh = sl → ordering not preserved
    assert result["verified"] is False


# ── verify_invariant_persistence — bounded ────────────────────────────────────

def test_bounded_verified_for_finite_category():
    """A small finite category always yields bounded hom-set profiles."""
    nn, lp, F, sh, sl, sa, lt, lf = build_nn_lp_scenario()
    checker = YonedaChecker()
    result = checker.verify_invariant_persistence(
        source_cat=nn, target_cat=lp, functor=F,
        invariant_type="bounded", objects=[sh]
    )
    assert result["verified"] is True
    assert any("VERIFIED" in step for step in result["proof_steps"])


def test_unknown_invariant_type_not_verified():
    """Unknown invariant type should return verified=False with an explanatory step."""
    nn, lp, F, sh, sl, *_ = build_nn_lp_scenario()
    checker = YonedaChecker()
    result = checker.verify_invariant_persistence(
        source_cat=nn, target_cat=lp, functor=F,
        invariant_type="recurrence_xyz", objects=[sh]
    )
    assert result["verified"] is False
    assert any("Unknown" in step for step in result["proof_steps"])


# ── proof_steps content ───────────────────────────────────────────────────────

def test_proof_steps_mention_functor_name():
    nn, lp, F, sh, sl, *_ = build_nn_lp_scenario()
    checker = YonedaChecker()
    result = checker.verify_invariant_persistence(
        source_cat=nn, target_cat=lp, functor=F,
        invariant_type="ordering", objects=[sh, sl]
    )
    assert any("NN_to_LP" in step for step in result["proof_steps"])


def test_proof_steps_contain_yoneda_profiles():
    nn, lp, F, sh, sl, *_ = build_nn_lp_scenario()
    checker = YonedaChecker()
    result = checker.verify_invariant_persistence(
        source_cat=nn, target_cat=lp, functor=F,
        invariant_type="ordering", objects=[sh, sl]
    )
    assert any("Yoneda profile" in step for step in result["proof_steps"])
