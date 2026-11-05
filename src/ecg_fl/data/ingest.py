"""MIT-BIH download, WFDB record loading, beat extraction and preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import wfdb
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ecg_fl.preprocessing.dsp import preprocess_ecg
from ecg_fl.preprocessing.segmentation import AAMI_CLASSES, extract_beats


@dataclass
class BeatDataset:
    x: np.ndarray
    y: np.ndarray
    records: np.ndarray
    subjects: np.ndarray
    raw_records: dict[str, tuple[np.ndarray, np.ndarray, int]]


def download_mitdb(root: str | Path, records: Iterable[str] | None = None) -> Path:
    """Download the official PhysioNet release; WFDB handles parsing/annotations."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # PhysioNet's unversioned landing page can stall behind redirects on some
    # networks. Pin the official release and provide the known record list so
    # WFDB downloads via its standard URL/file handling, without that lookup.
    official_records = [
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
        "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
        "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234",
    ]
    requested = [str(r).zfill(3) for r in records] if records is not None else official_records
    # WFDB resumes partial downloads by size; zero-byte fragments otherwise
    # pass an equality check against a failed empty response and remain corrupt.
    for partial in root.glob("*"):
        if partial.is_file() and partial.stat().st_size == 0:
            partial.unlink()
    base = "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0"
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, connect=4, read=4, backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
    )))
    for record in requested:
        for suffix in ("hea", "dat", "atr"):
            destination = root / f"{record}.{suffix}"
            if destination.exists() and destination.stat().st_size > 0:
                continue
            url = f"{base}/{record}.{suffix}"
            response = session.get(url, timeout=(20, 120), stream=True)
            if response.status_code == 404 and suffix == "dat":
                continue
            response.raise_for_status()
            with destination.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
            if destination.stat().st_size == 0:
                raise IOError(f"Empty response while downloading {url}")
    return root


def load_record(record_name: str, root: str | Path, lead: int = 0):
    """Load one WFDB record and its reference beat annotations."""
    record = wfdb.rdrecord(str(Path(root) / record_name), physical=True)
    annotation = wfdb.rdann(str(Path(root) / record_name), "atr")
    if record.p_signal is None or lead >= record.n_sig:
        raise ValueError(f"Record {record_name} has no requested signal lead {lead}")
    signal = record.p_signal[:, lead].astype(np.float32)
    return signal, np.asarray(annotation.sample), list(annotation.symbol), int(record.fs)


def build_dataset(
    root: str | Path,
    records: Iterable[str],
    *,
    lead: int = 0,
    before_seconds: float = 0.25,
    after_seconds: float = 0.45,
    dsp: bool = True,
    dsp_config: dict | None = None,
    strict: bool = False,
) -> BeatDataset:
    """Build labeled heartbeat arrays; failures are recorded or raised."""
    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_records: list[str] = []
    all_subjects: list[str] = []
    raw: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    failures: list[tuple[str, str]] = []
    for record_name in records:
        key = str(record_name).zfill(3)
        try:
            signal, samples, symbols, fs = load_record(key, root, lead)
            raw[key] = (signal.copy(), samples.copy(), fs)
            processed = preprocess_ecg(signal, fs, **(dsp_config or {})) if dsp else signal
            beats, labels, kept_samples = extract_beats(
                processed, samples, symbols, fs, before_seconds, after_seconds
            )
            all_x.extend(beats)
            all_y.extend(labels)
            all_records.extend([key] * len(labels))
            # MIT-BIH has 48 records from 47 subjects; records 201 and 202 belong to
            # the same subject, so preserve an explicit subject group identifier.
            subject = "201_202" if key in {"201", "202"} else key
            all_subjects.extend([subject] * len(labels))
        except Exception as exc:  # one corrupt/missing record must not hide others
            failures.append((key, repr(exc)))
            if strict:
                raise
    if failures:
        print(f"Skipped {len(failures)} records: {failures}")
    length = int(round((before_seconds + after_seconds) * 360))
    x = np.asarray(all_x, dtype=np.float32).reshape(-1, length)
    return BeatDataset(
        x=x,
        y=np.asarray(all_y, dtype=np.int64),
        records=np.asarray(all_records),
        subjects=np.asarray(all_subjects),
        raw_records=raw,
    )


def save_dataset(dataset: BeatDataset, path: str | Path) -> None:
    """Save only processed beats and labels; do not pickle raw signals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=dataset.x, y=dataset.y, records=dataset.records, subjects=dataset.subjects)


def load_dataset(path: str | Path) -> BeatDataset:
    with np.load(path, allow_pickle=False) as data:
        return BeatDataset(data["x"], data["y"], data["records"], data["subjects"], {})
