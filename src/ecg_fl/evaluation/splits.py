"""Leakage-safe grouped development folds for optional cross-validation."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_stratified_folds(
    labels: np.ndarray,
    subjects: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified folds that keep every subject in one fold.

    The function only partitions the provided development rows. Callers must
    exclude untouched test records before calling it.
    """
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(subjects).astype(str)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or not len(y):
        raise ValueError("labels and subjects must be aligned non-empty vectors")
    if n_splits < 2 or len(np.unique(groups)) < n_splits:
        raise ValueError("n_splits must be at least 2 and no greater than the subject count")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(train.astype(np.int64), valid.astype(np.int64))
             for train, valid in splitter.split(np.zeros(len(y)), y, groups)]
    validation_coverage = np.concatenate([valid for _, valid in folds])
    if not np.array_equal(np.sort(validation_coverage), np.arange(len(y))):
        raise RuntimeError("grouped folds did not validate each development row exactly once")
    for train, valid in folds:
        if set(groups[train]) & set(groups[valid]):
            raise RuntimeError("subject leakage detected between grouped fold train and validation")
    return folds
