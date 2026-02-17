"""Test that the Yoneda verification produces correct results for both functors."""
import pytest
from verification.categories import Category, Functor
from verification.yoneda_checker import YonedaChecker


def test_collapse_functor_destroys_ordering():
    """A functor that maps both objects to the same target should destroy ordering."""
    checker = YonedaChecker()

    nn_cat = Category("NN")
    s_low = nn_cat.add_object("low")
    s_high = nn_cat.add_object("high")
    nn_cat.add_morphism(s_low, s_high, "activate")

    lp_cat = Category("LP")
    lp_true = lp_cat.add_object("true")

    F = Functor("collapse", nn_cat, lp_cat)
    F.map_object(s_low, lp_true)
    F.map_object(s_high, lp_true)
    F.map_morphism(nn_cat.identity(s_low), lp_cat.identity(lp_true))
    F.map_morphism(nn_cat.identity(s_high), lp_cat.identity(lp_true))
    for m in nn_cat.morphisms:
        if m.source != m.target:
            F.map_morphism(m, lp_cat.identity(lp_true))

    result = checker.verify_invariant_persistence(
        nn_cat, lp_cat, F, "ordering", [s_high, s_low]
    )
    assert result["verified"] == False


def test_threshold_functor_preserves_ordering():
    """A proper threshold functor should preserve ordering."""
    checker = YonedaChecker()

    nn_cat = Category("NN")
    s_low = nn_cat.add_object("low")
    s_high = nn_cat.add_object("high")
    nn_cat.add_morphism(s_low, s_high, "activate")

    lp_cat = Category("LP")
    lp_false = lp_cat.add_object("false")
    lp_true = lp_cat.add_object("true")
    lp_assert = lp_cat.add_morphism(lp_false, lp_true, "assert")

    F = Functor("threshold", nn_cat, lp_cat)
    F.map_object(s_low, lp_false)
    F.map_object(s_high, lp_true)
    F.map_morphism(nn_cat.identity(s_low), lp_cat.identity(lp_false))
    F.map_morphism(nn_cat.identity(s_high), lp_cat.identity(lp_true))
    for m in nn_cat.morphisms:
        if m.source == s_low and m.target == s_high:
            F.map_morphism(m, lp_assert)

    result = checker.verify_invariant_persistence(
        nn_cat, lp_cat, F, "ordering", [s_high, s_low]
    )
    assert result["verified"] == True
