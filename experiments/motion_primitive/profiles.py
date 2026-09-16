"""The sole public motion-primitive HAR-CGCD experiment profile.

Historical joint VQ/GRU profiles live in ``legacy_profiles`` and are not
accepted by the public launchers.
"""

from __future__ import annotations

from typing import Iterable


PROFILE = "frozen_a2_e0_state_k32"
PROFILES = (PROFILE,)


def normalize_profile(value: str) -> str:
    """Return the strict public profile and reject legacy/unknown arms."""

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
            f"Duplicate public profiles are forbidden: {canonical}."
        )
    return canonical


__all__ = [
    "PROFILE",
    "PROFILES",
    "normalize_profile",
    "normalize_profile_grid",
]
