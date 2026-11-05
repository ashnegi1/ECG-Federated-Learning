"""Training-only class-weight calculation shared by central and FL paths."""
from __future__ import annotations

import numpy as np

from ecg_fl.preprocessing.segmentation import AAMI_CLASSES


def compute_class_weights(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inverse-frequency weights and counts in canonical AAMI order.

    Weights follow ``n / (K * count[class])`` over the supplied *training*
    labels. A class absent from the global training set gets weight zero rather
    than infinity; it cannot contribute positive training examples in that run.
    Callers must compute this once on the global training split and share the
    result with all clients.
    """
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if np.any((y < 0) | (y >= len(AAMI_CLASSES))):
        raise ValueError(f"labels must be integer indices in [0, {len(AAMI_CLASSES) - 1}]")
    counts = np.bincount(y, minlength=len(AAMI_CLASSES)).astype(np.int64)
    weights = np.zeros(len(AAMI_CLASSES), dtype=np.float32)
    present = counts > 0
    weights[present] = y.size / (len(AAMI_CLASSES) * counts[present])
    return weights, counts
