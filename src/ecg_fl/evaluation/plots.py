"""Plot actual training curves and controlled DSP result comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(central_metrics: dict, federated_metrics: dict, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    central_train = central_metrics.get("train_loss_by_epoch", [])
    central_val = central_metrics.get("validation_loss_by_epoch", [])
    fed_history = federated_metrics.get("training_history", {})
    if "train_metrics_by_round" in fed_history:
        fed_train = [float(v["train_loss"]) for _, v in sorted(fed_history["train_metrics_by_round"].items(), key=lambda item: int(item[0]))]
        fed_val = [float(v["validation_loss"]) for _, v in sorted(fed_history.get("validation_metrics_by_round", {}).items(), key=lambda item: int(item[0])) if "validation_loss" in v]
    else:
        fed_train = [v for _, v in fed_history.get("weighted_local_loss_by_round", [])]
        fed_val = [v for _, v in fed_history.get("validation_loss_by_round", [])]
    if central_train:
        ax.plot(range(1, len(central_train) + 1), central_train, marker="o", label="Central train")
    if central_val:
        ax.plot(range(1, len(central_val) + 1), central_val, marker="o", label="Central validation")
    if fed_train:
        ax.plot(range(1, len(fed_train) + 1), fed_train, marker="s", label="FedAvg weighted client train")
    if fed_val:
        ax.plot(range(1, len(fed_val) + 1), fed_val, marker="s", label="FedAvg global validation")
    ax.set(xlabel="Epoch / communication round", ylabel="Cross-entropy loss", title="Training convergence")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compare_dsp_runs(dsp_dir: str | Path, raw_dir: str | Path, output_path: str | Path) -> dict:
    """Write a paired descriptive comparison using real run metric files."""
    dsp_dir, raw_dir = Path(dsp_dir), Path(raw_dir)
    rows = {}
    for method, folder in (("dsp", dsp_dir), ("no_dsp", raw_dir)):
        rows[method] = {}
        for model in ("centralized", "federated"):
            metrics = json.loads((folder / f"{model}_metrics.json").read_text(encoding="utf-8"))
            rows[method][model] = {
                key: metrics[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
            }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels, values = [], []
    for method in ("dsp", "no_dsp"):
        for model in ("centralized", "federated"):
            labels.append(f"{method}\n{model}")
            values.append(rows[method][model]["f1_macro"])
    bars = ax.bar(labels, values, color=["#2b6cb0", "#63b3ed", "#805ad5", "#b794f4"])
    ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set(ylabel="Macro F1", title="DSP controlled smoke comparison (one epoch / round)", ylim=(0, 1))
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)
    return rows


def plot_preprocessing_example(raw: np.ndarray, processed: np.ndarray, fs: int, path: str | Path) -> None:
    """Save a raw-versus-Butterworth+DWT ECG trace for demonstrations."""
    count = min(len(raw), len(processed), int(10 * fs))
    t = np.arange(count) / fs
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, raw[:count], label="Raw", alpha=.6, linewidth=.8)
    ax.plot(t, processed[:count], label="Butterworth 0.5–45 Hz + DWT", linewidth=.9)
    ax.set(xlabel="Time (s)", ylabel="ECG (mV)", title="MIT-BIH record preprocessing example")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_class_distribution(labels: np.ndarray, path: str | Path) -> None:
    from ecg_fl.preprocessing.segmentation import AAMI_CLASSES
    counts = np.bincount(labels, minlength=len(AAMI_CLASSES))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(AAMI_CLASSES, counts, color="#3b82a0")
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in counts], padding=2)
    ax.set(ylabel="Beat count", title="AAMI class distribution")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
