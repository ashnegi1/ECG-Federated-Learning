"""Current Flower Simulation Runtime wrapper for the ECG Flower Apps."""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import torch
from flwr.simulation import run_simulation

from ecg_fl.data.partition import partition_by_record
from ecg_fl.models.cnn import ECGCNN
from ecg_fl.federated.model_state import SUPPORTED_BATCHNORM_POLICIES


def run_federated(
    x: np.ndarray,
    y: np.ndarray,
    partitions: list[np.ndarray],
    *,
    rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    output_dir: str | Path = "outputs/federated",
    seed: int = 42,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    class_weighting: bool = False,
    class_weights: np.ndarray | list[float] | None = None,
    batchnorm_policy: str = "fedavg_full_state",
    federated_strategy: str = "fedavg",
    proximal_mu: float = 0.01,
    partition_metadata: list[dict] | None = None,
) -> tuple[ECGCNN, dict]:
    """Run Flower ClientApp/ServerApp with FedAvg or the FedProx local objective.

    The centralized coordinator writes disjoint client partitions to local files
    before simulation. Each Flower ClientApp loads only its own file. The server
    receives model arrays and scalar metrics, plus it evaluates global parameters
    on the independent validation archive.
    """
    if validation_data is None or len(validation_data[1]) == 0:
        raise ValueError("Independent validation data is required for global evaluation")
    if batchnorm_policy not in SUPPORTED_BATCHNORM_POLICIES:
        raise ValueError(f"unsupported BatchNorm aggregation policy: {batchnorm_policy}")
    if federated_strategy not in {"fedavg", "fedprox"}:
        raise ValueError("federated_strategy must be 'fedavg' or 'fedprox'")
    if not np.isfinite(proximal_mu) or proximal_mu < 0:
        raise ValueError("proximal_mu must be finite and non-negative")
    if class_weighting:
        if class_weights is None:
            raise ValueError("global class_weights are required when class_weighting is enabled")
        class_weights = np.asarray(class_weights, dtype=np.float32)
        if class_weights.shape != (5,) or not np.isfinite(class_weights).all() or np.any(class_weights < 0):
            raise ValueError("class_weights must contain five finite, non-negative values")
    active = [idx for idx in partitions if len(idx)]
    if len(active) != len(partitions):
        raise ValueError("Every configured Flower client must have at least one beat")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client_dir = output_dir / "client_data"
    client_dir.mkdir(parents=True, exist_ok=True)
    for client_id, indices in enumerate(active):
        np.savez_compressed(client_dir / f"partition_{client_id}.npz", x=x[indices], y=y[indices])
    validation_path = output_dir / "validation.npz"
    np.savez_compressed(validation_path, x=validation_data[0], y=validation_data[1])
    os.environ["ECG_CLIENT_PARTITION_DIR"] = str(client_dir)
    os.environ["ECG_VALIDATION_ARCHIVE"] = str(validation_path)
    os.environ["ECG_FL_OUTPUT_DIR"] = str(output_dir)
    os.environ["ECG_FL_ROUNDS"] = str(rounds)
    os.environ["ECG_FL_CLIENTS"] = str(len(active))
    os.environ["ECG_LOCAL_EPOCHS"] = str(local_epochs)
    os.environ["ECG_BATCH_SIZE"] = str(batch_size)
    os.environ["ECG_LEARNING_RATE"] = str(learning_rate)
    os.environ["ECG_FL_SEED"] = str(seed)
    os.environ["ECG_CLASS_WEIGHTING"] = "1" if class_weighting else "0"
    os.environ["ECG_CLASS_WEIGHTS"] = json.dumps(
        [] if not class_weighting else list(map(float, class_weights))
    )
    os.environ["ECG_BATCHNORM_POLICY"] = batchnorm_policy
    os.environ["ECG_FL_STRATEGY"] = federated_strategy
    os.environ["ECG_PROXIMAL_MU"] = str(proximal_mu)
    if partition_metadata is not None:
        (output_dir / "partition_metadata.json").write_text(
            json.dumps(partition_metadata, indent=2), encoding="utf-8"
        )
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import app objects after environment paths are set so Ray workers inherit
    # exactly the local simulated-client file locations.
    from ecg_fl.federated.client_app import app as client_app
    from ecg_fl.federated.server_app import app as server_app

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(active),
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )
    model = ECGCNN(input_length=x.shape[1])
    model.load_state_dict(torch.load(output_dir / "federated_model_state.pt", map_location="cpu", weights_only=True))
    history = json.loads((output_dir / "flower_history.json").read_text(encoding="utf-8"))
    return model, history
