import pytest
from verification.categories import Object, Morphism, Category, Functor, NaturalTransformation


# ── helpers ──────────────────────────────────────────────────────────────────

def two_object_category(name="C"):
    """Return a category with objects A, B and a morphism A→B."""
    cat = Category(name)
    a = cat.add_object("A")
    b = cat.add_object("B")
    f = cat.add_morphism(a, b, "f")
    return cat, a, b, f


# ── Object / Morphism ────────────────────────────────────────────────────────

def test_object_hashable_and_eq():
    o1 = Object("x", "C")
    o2 = Object("x", "C")
    assert o1 == o2
    assert hash(o1) == hash(o2)


def test_morphism_hashable():
    a = Object("A", "C")
    b = Object("B", "C")
    m = Morphism(a, b, "f")
    assert hash(m) is not None


# ── Category ─────────────────────────────────────────────────────────────────

def test_add_object_creates_identity():
    cat = Category("C")
    a = cat.add_object("A")
    assert a in cat.objects
    id_a = cat.identity(a)
    assert id_a.source == a
    assert id_a.target == a


def test_add_morphism_stored():
    cat, a, b, f = two_object_category()
    assert f in cat.morphisms
    assert f.source == a
    assert f.target == b


def test_default_morphism_name():
    cat = Category("C")
    a = cat.add_object("A")
    b = cat.add_object("B")
    m = cat.add_morphism(a, b)
    assert "A" in m.name and "B" in m.name


def test_compose_returns_morphism():
    cat = Category("C")
    a = cat.add_object("A")
    b = cat.add_object("B")
    c = cat.add_object("C")
    f = cat.add_morphism(a, b, "f")
    g = cat.add_morphism(b, c, "g")
    gf = cat.compose(f, g)
    assert gf is not None
    assert gf.source == a
    assert gf.target == c


def test_compose_non_composable_returns_none():
    cat = Category("C")
    a = cat.add_object("A")
    b = cat.add_object("B")
    c = cat.add_object("C")
    f = cat.add_morphism(a, b, "f")
    g = cat.add_morphism(a, c, "g")  # same source — not composable after f
    assert cat.compose(f, g) is None


def test_compose_cached():
    cat = Category("C")
    a = cat.add_object("A")
    b = cat.add_object("B")
    c = cat.add_object("C")
    f = cat.add_morphism(a, b, "f")
    g = cat.add_morphism(b, c, "g")
    gf1 = cat.compose(f, g)
    gf2 = cat.compose(f, g)
    assert gf1 is gf2


def test_hom_set():
    cat, a, b, f = two_object_category()
    hom = cat.hom_set(a, b)
    assert f in hom


def test_hom_set_excludes_wrong_direction():
    cat, a, b, f = two_object_category()
    hom_ba = cat.hom_set(b, a)
    assert f not in hom_ba


# ── Functor ──────────────────────────────────────────────────────────────────

def test_functor_apply_object():
    src = Category("Src")
    tgt = Category("Tgt")
    a = src.add_object("A")
    x = tgt.add_object("X")
    F = Functor("F", src, tgt)
    F.map_object(a, x)
    assert F.apply_object(a) == x


def test_functor_apply_morphism():
    src = Category("Src")
    tgt = Category("Tgt")
    a = src.add_object("A")
    b = src.add_object("B")
    x = tgt.add_object("X")
    y = tgt.add_object("Y")
    f = src.add_morphism(a, b, "f")
    g = tgt.add_morphism(x, y, "g")
    F = Functor("F", src, tgt)
    F.map_morphism(f, g)
    assert F.apply_morphism(f) == g


def test_functor_faithful_when_injective():
    src = Category("Src")
    tgt = Category("Tgt")
    a = src.add_object("A")
    b = src.add_object("B")
    c = src.add_object("C")
    x = tgt.add_object("X")
    y = tgt.add_object("Y")
    z = tgt.add_object("Z")
    f = src.add_morphism(a, b, "f")
    g = src.add_morphism(b, c, "g")
    fx = tgt.add_morphism(x, y, "fx")
    gy = tgt.add_morphism(y, z, "gy")
    F = Functor("F", src, tgt)
    F.map_morphism(f, fx)
    F.map_morphism(g, gy)
    assert F.is_faithful() is True


def test_functor_not_faithful_when_collapsing():
    src = Category("Src")
    tgt = Category("Tgt")
    a = src.add_object("A")
    b = src.add_object("B")
    c = src.add_object("C")
    x = tgt.add_object("X")
    y = tgt.add_object("Y")
    f = src.add_morphism(a, b, "f")
    g = src.add_morphism(a, c, "g")
    h = tgt.add_morphism(x, y, "h")
    F = Functor("F", src, tgt)
    F.map_morphism(f, h)
    F.map_morphism(g, h)   # both map to same target — not faithful
    assert F.is_faithful() is False


def test_functor_preserves_composition():
    src = Category("Src")
    tgt = Category("Tgt")
    a = src.add_object("A")
    b = src.add_object("B")
    c = src.add_object("C")
    x = tgt.add_object("X")
    y = tgt.add_object("Y")
    z = tgt.add_object("Z")
    f = src.add_morphism(a, b, "f")
    g = src.add_morphism(b, c, "g")
    gf = src.compose(f, g)
    fx = tgt.add_morphism(x, y, "fx")
    gy = tgt.add_morphism(y, z, "gy")
    gfx = tgt.compose(fx, gy)
    F = Functor("F", src, tgt)
    F.map_morphism(f, fx)
    F.map_morphism(g, gy)
    F.map_morphism(gf, gfx)
    assert F.preserves_composition() is True


# ── NaturalTransformation ────────────────────────────────────────────────────

def test_natural_transformation_mismatched_sources_raises():
    c1 = Category("C1")
    c2 = Category("C2")
    d = Category("D")
    F = Functor("F", c1, d)
    G = Functor("G", c2, d)
    with pytest.raises(AssertionError):
        NaturalTransformation("eta", F, G)


def test_naturality_trivially_holds_with_no_mapped_morphisms():
    """When no morphisms have functor images, no squares can fail."""
    c = Category("C")
    d = Category("D")
    a = c.add_object("A")
    b = c.add_object("B")
    x = d.add_object("X")
    y = d.add_object("Y")
    F = Functor("F", c, d)
    G = Functor("G", c, d)
    F.map_object(a, x)
    G.map_object(a, x)
    m_xy = d.add_morphism(x, y, "m")
    eta = NaturalTransformation("eta", F, G)
    eta.set_component(a, m_xy)
    ok, violations = eta.check_naturality()
    assert ok is True
    assert violations == []
