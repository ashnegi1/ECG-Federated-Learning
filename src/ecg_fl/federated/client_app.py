"""Current Flower ClientApp: clients train on their local beat archive."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict

from ecg_fl.models.cnn import ECGCNN
from ecg_fl.federated.auditable_fedavg import FEDAVG, FEDPROX, SUPPORTED_FEDERATED_STRATEGIES
from ecg_fl.federated.model_state import FEDAVG_FULL_STATE, SUPPORTED_BATCHNORM_POLICIES, model_state_for_fedavg
from ecg_fl.training.engine import train_model

app = ClientApp()


def load_partition(partition_id: int) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(os.environ.get("ECG_CLIENT_PARTITION_DIR", "data/flower_clients"))
    path = data_dir / f"partition_{partition_id}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Client partition {partition_id} not found at {path}. "
            "Run the main pipeline so record-safe client shards are prepared first."
        )
    with np.load(path, allow_pickle=False) as data:
        return data["x"].astype(np.float32, copy=False), data["y"].astype(np.int64, copy=False)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Receive global arrays, locally train, and return arrays plus scalar metrics."""
    partition_id = int(context.node_config["partition-id"])
    seed = int(msg.content["config"].get("seed", 42))
    torch.manual_seed(seed + partition_id)
    np.random.seed(seed + partition_id)
    x, y = load_partition(partition_id)
    config = msg.content["config"]
    batchnorm_policy = str(config.get("batchnorm-policy", FEDAVG_FULL_STATE))
    if batchnorm_policy not in SUPPORTED_BATCHNORM_POLICIES:
        raise ValueError(f"unsupported BatchNorm aggregation policy: {batchnorm_policy}")
    model = ECGCNN(input_length=x.shape[1])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    strategy_name = str(config.get("federated-strategy", FEDAVG)).lower()
    if strategy_name not in SUPPORTED_FEDERATED_STRATEGIES:
        raise ValueError(f"unsupported federated strategy: {strategy_name}")
    proximal_mu = float(config.get("proximal-mu", 0.0)) if strategy_name == FEDPROX else 0.0
    # Snapshot the received global trainable parameters before local updates.
    proximal_reference = (
        {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        if strategy_name == FEDPROX else None
    )
    losses = train_model(
        model,
        x,
        y,
        epochs=int(config["local-epochs"]),
        batch_size=int(config["batch-size"]),
        learning_rate=float(config["learning-rate"]),
        weight_decay=float(config["weight-decay"]),
        class_weighting=bool(config["class-weighting"]),
        class_weights=(
            [float(config[f"class-weight-{i}"]) for i in range(5)]
            if bool(config["class-weighting"]) else None
        ),
        device=torch.device("cpu"),
        proximal_mu=proximal_mu,
        proximal_reference=proximal_reference,
    )
    content = RecordDict(
        {
            "arrays": ArrayRecord(model_state_for_fedavg(model, batchnorm_policy)),
            "metrics": MetricRecord(
                {"train_loss": float(losses[-1]), "num-examples": int(len(y))}
            ),
        }
    )
    return Message(content=content, reply_to=msg)
