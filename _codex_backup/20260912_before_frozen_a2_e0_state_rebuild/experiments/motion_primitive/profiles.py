"""Canonical profile name for the motion-primitive HAR-CGCD project.

Trial pooling is deliberately not an HHR experiment arm.  Historical pooling
results remain external references; accepting them here would make it possible
to launch a non-primitive pipeline from the HHR entry points by accident.
"""

from __future__ import annotations

from typing import Iterable


PROFILE_JOINT = "motion_primitive_joint"
PROFILES = (PROFILE_JOINT,)


def normalize_profile(value: str) -> str:
    """Return the canonical HHR profile name and reject every other arm."""

    canonical = str(value).strip()
    if canonical not in PROFILES:
        raise ValueError(f"Unknown profile {value!r}; expected one of {PROFILES}.")
    return canonical


def normalize_profile_grid(values: Iterable[str]) -> tuple[str, ...]:
    """Canonicalise a grid and reject aliases that collapse to duplicates."""

    canonical = tuple(normalize_profile(value) for value in values)
    if not canonical:
        raise ValueError("The profile grid cannot be empty.")
    if len(canonical) != len(set(canonical)):
        raise ValueError(
            "Duplicate profiles are forbidden after legacy-name normalisation: "
            f"{canonical}."
        )
    return canonical


__all__ = [
    "PROFILE_JOINT",
    "PROFILES",
    "normalize_profile",
    "normalize_profile_grid",
]
