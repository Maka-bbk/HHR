"""Motion-primitive trajectory feasibility experiment."""

from .core import (
    association_summary,
    build_trial_sequence,
    cross_subject_pair_indices,
    distance_matrix_from_histograms,
    distance_matrix_from_sequences,
    fit_weighted_pca,
    normalized_levenshtein,
    run_length_encode,
)

__all__ = [
    "association_summary",
    "build_trial_sequence",
    "cross_subject_pair_indices",
    "distance_matrix_from_histograms",
    "distance_matrix_from_sequences",
    "fit_weighted_pca",
    "normalized_levenshtein",
    "run_length_encode",
]

