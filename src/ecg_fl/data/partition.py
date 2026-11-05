"""Seeded, record/subject-safe client allocation and partition reporting."""
from __future__ import annotations

import numpy as np

from ecg_fl.preprocessing.segmentation import AAMI_CLASSES


def partition_by_record(
    labels: np.ndarray,
    records: np.ndarray,
    num_clients: int,
    alpha: float = 0.3,
    seed: int = 42,
    subjects: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Assign whole subject groups to clients while balancing total beat load.

    Records belonging to the same subject are indivisible. A seeded greedy
    largest-group-first allocator minimizes avoidable client-size imbalance.
    Groups containing classes represented by at most ``num_clients`` subject
    groups are allocated to clients missing those classes when possible. This
    spreads rare *records* where the dataset permits it without splitting or
    duplicating a beat. Remaining class skew is retained because records, not
    individual examples, are the allocation units.

    ``alpha`` controls the seeded slack around the least-loaded eligible client:
    zero is strict greedy balancing; larger values permit more assignment
    variation and client-size skew while preserving deterministic output.
    """
    y = np.asarray(labels, dtype=np.int64)
    rec = np.asarray(records).astype(str)
    groups = rec if subjects is None else np.asarray(subjects).astype(str)
    if y.ndim != 1 or rec.ndim != 1 or groups.ndim != 1 or not (len(y) == len(rec) == len(groups)):
        raise ValueError("labels, records, and subjects must be aligned one-dimensional arrays")
    if not len(y):
        raise ValueError("cannot partition an empty dataset")
    if num_clients < 2 or alpha < 0:
        raise ValueError("num_clients must be >= 2 and alpha must be >= 0")
    if np.any((y < 0) | (y >= len(AAMI_CLASSES))):
        raise ValueError("labels are outside the five AAMI class indices")

    unique_groups = np.unique(groups)
    if len(unique_groups) < num_clients:
        raise ValueError("There must be at least one subject group per client")

    rng = np.random.default_rng(seed)
    group_rows: list[dict] = []
    class_group_support = np.zeros(len(AAMI_CLASSES), dtype=np.int64)
    for group in unique_groups:
        indices = np.flatnonzero(groups == group)
        counts = np.bincount(y[indices], minlength=len(AAMI_CLASSES))
        class_group_support += counts > 0
        group_rows.append({
            "group": group,
            "indices": indices,
            "size": len(indices),
            "counts": counts,
        })

    def priority(row: dict) -> float:
        present = np.flatnonzero(row["counts"] > 0)
        score = 0.0
        for class_id in present:
            support = class_group_support[class_id]
            if 0 < support <= num_clients:
                score += 1.0 / float(support)
        return score

    # Shuffle before stable sorting to make ties seeded rather than dependent
    # on source-record ordering.
    rng.shuffle(group_rows)
    group_rows.sort(key=lambda row: (priority(row), row["size"]), reverse=True)

    assignments: list[list[int]] = [[] for _ in range(num_clients)]
    client_sizes = np.zeros(num_clients, dtype=np.int64)
    client_class_groups = np.zeros((num_clients, len(AAMI_CLASSES)), dtype=np.int64)
    target_size = len(y) / float(num_clients)

    for row in group_rows:
        rare_classes = [
            c for c in range(len(AAMI_CLASSES))
            if row["counts"][c] > 0 and 0 < class_group_support[c] <= num_clients
        ]
        projected = client_sizes + row["size"]
        if rare_classes:
            uncovered = np.all(client_class_groups[:, rare_classes] == 0, axis=1)
            if uncovered.any():
                eligible = np.flatnonzero(uncovered)
                projected_eligible = projected[eligible]
                min_projected = int(projected_eligible.min())
                slack = alpha * target_size
                eligible = eligible[projected_eligible <= min_projected + slack]
            else:
                eligible = np.arange(num_clients)
        else:
            eligible = np.arange(num_clients)

        min_projected = int(projected[eligible].min())
        slack = alpha * target_size
        near_min = eligible[projected[eligible] <= min_projected + slack]
        client_id = int(rng.choice(near_min)) if len(near_min) > 1 else int(near_min[0])
        assignments[client_id].append(row["group"])
        client_sizes[client_id] += row["size"]
        client_class_groups[client_id] += (row["counts"] > 0).astype(np.int64)

    result = [np.flatnonzero(np.isin(groups, client_groups)) for client_groups in assignments]
    if any(len(indices) == 0 for indices in result):
        raise RuntimeError("partition allocator produced an empty client")
    return result


def describe_partitions(
    labels: np.ndarray,
    records: np.ndarray,
    subjects: np.ndarray,
    partitions: list[np.ndarray],
) -> list[dict]:
    """Return JSON-ready assignment, class, and FedAvg size-weight metadata."""
    y = np.asarray(labels, dtype=np.int64)
    rec = np.asarray(records).astype(str)
    subj = np.asarray(subjects).astype(str)
    total = sum(len(indices) for indices in partitions)
    if len(y) != len(rec) or len(y) != len(subj) or total == 0:
        raise ValueError("partition metadata inputs must be aligned and non-empty")
    rows = []
    for client_id, indices in enumerate(partitions):
        counts = np.bincount(y[indices], minlength=len(AAMI_CLASSES))
        n = int(len(indices))
        rows.append({
            "client_id": client_id,
            "sample_count": n,
            "records": sorted(set(rec[indices].tolist())),
            "subjects": sorted(set(subj[indices].tolist())),
            "class_counts": {name: int(counts[i]) for i, name in enumerate(AAMI_CLASSES)},
            "class_proportions": {
                name: (float(counts[i] / n) if n else 0.0)
                for i, name in enumerate(AAMI_CLASSES)
            },
            "absent_classes": [name for i, name in enumerate(AAMI_CLASSES) if counts[i] == 0],
            "fedavg_sample_weight": float(n / total),
        })
    return rows


def assert_group_isolation(
    records: np.ndarray,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
) -> None:
    """Raise when rows, records, or subjects overlap between named splits."""
    rec = np.asarray(records).astype(str)
    subj = np.asarray(subjects).astype(str)
    owners: dict[str, dict[str, set]] = {"records": {}, "subjects": {}, "rows": {}}
    for split_name, indices in split_indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        if np.any((idx < 0) | (idx >= len(rec))):
            raise ValueError(f"split {split_name!r} contains an out-of-range row index")
        owners["records"][split_name] = set(rec[idx].tolist())
        owners["subjects"][split_name] = set(subj[idx].tolist())
        owners["rows"][split_name] = set(map(int, idx.tolist()))
    names = list(split_indices)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for unit in ("rows", "records", "subjects"):
                overlap = owners[unit][left] & owners[unit][right]
                if overlap:
                    raise ValueError(
                        f"{unit} leakage between {left!r} and {right!r}: {sorted(overlap)[:5]}"
                    )
