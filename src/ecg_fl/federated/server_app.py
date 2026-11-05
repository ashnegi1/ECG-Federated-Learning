"""Current Flower ServerApp coordinating Flower FedAvg rounds."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from flwr.common import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import ServerApp

from ecg_fl.federated.auditable_fedavg import (
    FEDAVG, FEDPROX, make_federated_strategy,
)
from ecg_fl.federated.model_state import FEDAVG_FULL_STATE, SUPPORTED_BATCHNORM_POLICIES, model_state_for_fedavg
from ecg_fl.models.cnn import ECGCNN
from ecg_fl.preprocessing.segmentation import AAMI_CLASSES
from ecg_fl.training.engine import evaluate_loss

app = ServerApp()


def evaluate_global(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate global parameters on the held-out validation records only."""
    validation_path = Path(os.environ.get("ECG_VALIDATION_ARCHIVE", "data/flower_validation.npz"))
    with np.load(validation_path, allow_pickle=False) as data:
        x_val, y_val = data["x"], data["y"]
    model = ECGCNN(input_length=x_val.shape[1])
    model.load_state_dict(arrays.to_torch_state_dict(), strict=True)
    loss = evaluate_loss(model, x_val, y_val, device=torch.device("cpu"))
    support = np.bincount(y_val, minlength=len(AAMI_CLASSES))
    metrics = {"validation_loss": float(loss), "num-examples": int(len(y_val))}
    metrics.update({f"support_{name}": int(support[i]) for i, name in enumerate(AAMI_CLASSES)})
    return MetricRecord(metrics)


def _plain_metrics(metrics_by_round: dict) -> dict:
    result = {}
    for rnd, record in metrics_by_round.items():
        result[str(rnd)] = dict(record)
    return result


def _load_run_metadata(run_config: ConfigRecord) -> dict:
    """Load shared training-only class weights for direct Flower CLI runs."""
    metadata_path = Path(os.environ.get("ECG_RUN_METADATA", "data/flower_run_metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if "ECG_CLASS_WEIGHTING" in os.environ:
        enabled_value = os.environ["ECG_CLASS_WEIGHTING"]
    elif "class_weighting" in metadata:
        enabled_value = metadata["class_weighting"]
    else:
        enabled_value = run_config.get("class-weighting", True)
    enabled = enabled_value if isinstance(enabled_value, bool) else str(enabled_value).lower() in {"1", "true", "yes"}
    env_weights = os.environ.get("ECG_CLASS_WEIGHTS")
    weights = json.loads(env_weights) if env_weights is not None else metadata.get("class_weights")
    if enabled and (weights is None or len(weights) != 5):
        raise ValueError(
            "global class weights are missing; run `ecg-fl prepare-flower` or the main `ecg-fl run` first"
        )
    policy = str(os.environ.get(
        "ECG_BATCHNORM_POLICY",
        metadata.get("batchnorm_policy", run_config.get("batchnorm-policy", FEDAVG_FULL_STATE)),
    ))
    if policy not in SUPPORTED_BATCHNORM_POLICIES:
        raise ValueError(f"unsupported BatchNorm aggregation policy: {policy}")
    strategy_name = str(os.environ.get(
        "ECG_FL_STRATEGY", metadata.get("federated_strategy", run_config.get("strategy", FEDAVG))
    )).lower()
    proximal_mu = float(os.environ.get(
        "ECG_PROXIMAL_MU", metadata.get("proximal_mu", run_config.get("proximal-mu", 0.01))
    ))
    if not np.isfinite(proximal_mu) or proximal_mu < 0:
        raise ValueError("proximal-mu must be finite and non-negative")
    return {
        "class_weighting": enabled, "class_weights": weights,
        "batchnorm_policy": policy, "federated_strategy": strategy_name,
        "proximal_mu": proximal_mu,
    }


@app.main()
def main(grid, context: Context) -> None:
    """Create initial model, run Flower's current FedAvg strategy and save outputs."""
    run_config = context.run_config
    rounds = int(run_config.get("num-server-rounds", os.environ.get("ECG_FL_ROUNDS", 5)))
    clients = int(run_config.get("num-clients", os.environ.get("ECG_FL_CLIENTS", 5)))
    seed = int(run_config.get("seed", os.environ.get("ECG_FL_SEED", 42)))
    run_metadata = _load_run_metadata(run_config)
    torch.manual_seed(seed)
    model = ECGCNN(input_length=int(run_config.get("input-length", 252)))
    initial_arrays = ArrayRecord(model_state_for_fedavg(model, run_metadata["batchnorm_policy"]))
    strategy = make_federated_strategy(
        run_metadata["federated_strategy"],
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=clients,
        min_evaluate_nodes=0,
        min_available_nodes=clients,
        batchnorm_policy=run_metadata["batchnorm_policy"],
    )
    train_config = {
        "local-epochs": int(run_config.get("local-epochs", os.environ.get("ECG_LOCAL_EPOCHS", 1))),
        "batch-size": int(run_config.get("batch-size", os.environ.get("ECG_BATCH_SIZE", 128))),
        "learning-rate": float(run_config.get("learning-rate", os.environ.get("ECG_LEARNING_RATE", 0.001))),
        "weight-decay": float(run_config.get("weight-decay", 0.0001)),
        "class-weighting": run_metadata["class_weighting"],
        "batchnorm-policy": run_metadata["batchnorm_policy"],
        "federated-strategy": run_metadata["federated_strategy"],
        "proximal-mu": run_metadata["proximal_mu"],
        "seed": seed,
    }
    if run_metadata["class_weighting"]:
        for i, value in enumerate(run_metadata["class_weights"]):
            train_config[f"class-weight-{i}"] = float(value)
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord(train_config),
        num_rounds=rounds,
        evaluate_fn=evaluate_global,
    )
    output_dir = Path(os.environ.get("ECG_FL_OUTPUT_DIR", "outputs/federated"))
    output_dir.mkdir(parents=True, exist_ok=True)
    state = result.arrays.to_torch_state_dict()
    torch.save(state, output_dir / "federated_model.pt")
    torch.save(state, output_dir / "federated_model_state.pt")
    history = {
        "rounds": rounds,
        "clients": clients,
        "train_metrics_by_round": _plain_metrics(result.train_metrics_clientapp),
        "validation_metrics_by_round": _plain_metrics(result.evaluate_metrics_serverapp),
        "class_weighting": run_metadata["class_weighting"],
        "class_weights": run_metadata["class_weights"] if run_metadata["class_weighting"] else None,
        "batchnorm_policy": run_metadata["batchnorm_policy"],
        "federated_strategy": run_metadata["federated_strategy"],
        "proximal_mu": run_metadata["proximal_mu"],
        "round_details": strategy.round_audit,
    }
    for round_key, detail in history["round_details"].items():
        validation = history["validation_metrics_by_round"].get(round_key, {})
        detail["global_validation_loss"] = validation.get("validation_loss")
    (output_dir / "flower_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
