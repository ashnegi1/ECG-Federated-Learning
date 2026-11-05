"""Classification metrics and figures."""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from ecg_fl.preprocessing.segmentation import AAMI_CLASSES
from ecg_fl.training.engine import predict


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(AAMI_CLASSES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=AAMI_CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": list(AAMI_CLASSES),
    }


def evaluate_test_pair(
    centralized_model,
    federated_model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict]:
    """Evaluate both models on the same immutable test arrays and labels."""
    y_true = np.asarray(y_test)
    x_shared = np.asarray(x_test)
    return {
        "centralized": classification_metrics(y_true, predict(centralized_model, x_shared)),
        "federated": classification_metrics(y_true, predict(federated_model, x_shared)),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_confusion(metrics: dict, path: str | Path) -> None:
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=AAMI_CLASSES, yticklabels=AAMI_CLASSES,
           xlabel="Predicted", ylabel="True", title="MIT-BIH beat classification")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
