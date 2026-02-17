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
        s1 = nn_cat.add_object("state_high")    # NN output >= 0.5
        nn_cat.add_morphism(s0, s1, "activate") # Transition low → high
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
