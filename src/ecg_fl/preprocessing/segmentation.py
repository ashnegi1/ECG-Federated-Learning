"""R-peak centered beat extraction and explicit AAMI label mapping."""
from __future__ import annotations

import numpy as np

AAMI_CLASSES = ("N", "S", "V", "F", "Q")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(AAMI_CLASSES)}
ANNOTATION_TO_AAMI = {
    **{s: "N" for s in ("N", "L", "R", "e", "j")},
    **{s: "S" for s in ("A", "a", "J", "S")},
    **{s: "V" for s in ("V", "E")},
    "F": "F",
    **{s: "Q" for s in ("/", "f", "Q")},
}


def extract_beats(
    signal: np.ndarray,
    samples: np.ndarray,
    symbols: list[str],
    fs: int,
    before_seconds: float = 0.25,
    after_seconds: float = 0.45,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    before = int(round(before_seconds * fs))
    after = int(round(after_seconds * fs))
    beats: list[np.ndarray] = []
    labels: list[int] = []
    kept: list[int] = []
    if len(samples) != len(symbols):
        raise ValueError("Annotation samples and symbols must have equal length")
    for sample, symbol in zip(samples, symbols):
        aami = ANNOTATION_TO_AAMI.get(symbol)
        start, end = int(sample) - before, int(sample) + after
        if aami is None or start < 0 or end > len(signal):
            continue
        beat = np.asarray(signal[start:end], dtype=np.float32)
        if beat.size != before + after:
            continue
        beats.append(beat)
        labels.append(CLASS_TO_INDEX[aami])
        kept.append(int(sample))
    return beats, labels, kept
