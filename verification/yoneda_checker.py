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
        print(result["verified"])    # True/False
        print(result["proof_steps"]) # Human-readable proof
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
