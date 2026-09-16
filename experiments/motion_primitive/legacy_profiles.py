"""Legacy joint VQ/GRU profile registry.

This module exists only so historical joint-training runners and artifact
readers remain reproducible after the public HHR entry points move to the
frozen A2/E0/state/K32 route.  New experiments must use ``profiles.py`` and
the strict runners instead.
"""

from __future__ import annotations

from typing import Iterable


PROFILE_JOINT = "motion_primitive_joint"
PROFILES = (PROFILE_JOINT,)


def normalize_profile(value: str) -> str:
    canonical = str(value).strip()
    if canonical not in PROFILES:
        raise ValueError(f"Unknown profile {value!r}; expected one of {PROFILES}.")
    return canonical


def normalize_profile_grid(values: Iterable[str]) -> tuple[str, ...]:
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
