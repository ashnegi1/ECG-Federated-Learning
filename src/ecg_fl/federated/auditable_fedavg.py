"""FedAvg wrapper that persists selected, successful, and failed client updates."""
from __future__ import annotations

from typing import Iterable

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.server.grid import Grid
from flwr.serverapp.strategy import FedAvg
import torch
from ecg_fl.federated.model_state import FEDAVG_FULL_STATE, FEDAVG_POOLED_BN, SUPPORTED_BATCHNORM_POLICIES

FEDAVG = "fedavg"
FEDPROX = "fedprox"
SUPPORTED_FEDERATED_STRATEGIES = (FEDAVG, FEDPROX)


def _pool_batchnorm_buffers(aggregated: ArrayRecord, replies: list[Message], weighted_by_key: str) -> ArrayRecord:
    """Replace averaged BN EMA buffers with a sample-weighted moment pool.

    This moment-matching approximation combines clients' local running
    statistics. It is an experimental policy, not canonical FedAvg.
    """
    states = [(m.content["arrays"].to_torch_state_dict(), int(m.content["metrics"].get(weighted_by_key, 0)))
              for m in replies if not m.has_error()]
    total = sum(count for _, count in states)
    if total <= 0:
        return aggregated
    result = aggregated.to_torch_state_dict()
    for key in list(result):
        if not key.endswith("running_mean"):
            continue
        var_key = key[:-len("running_mean")] + "running_var"
        if var_key not in result:
            continue
        mean = sum(state[key].to(torch.float64) * count for state, count in states) / total
        variance = sum(
            (state[var_key].to(torch.float64) + (state[key].to(torch.float64) - mean) ** 2) * count
            for state, count in states
        ) / total
        result[key] = mean.to(result[key].dtype)
        result[var_key] = variance.to(result[var_key].dtype)
        counter_key = key[:-len("running_mean")] + "num_batches_tracked"
        if counter_key in result:
            result[counter_key] = torch.stack([state[counter_key] for state, _ in states]).max().to(result[counter_key].dtype)
    return ArrayRecord(result)


def summarize_round_replies(
    server_round: int,
    selected_node_ids: Iterable[int],
    replies: Iterable[Message],
    *,
    weighted_by_key: str = "num-examples",
) -> dict:
    """Build an auditable round record from actual Flower replies."""
    selected = {int(node_id) for node_id in selected_node_ids}
    messages = list(replies)
    successful = [m for m in messages if not m.has_error()]
    errored = [m for m in messages if m.has_error()]
    successful_ids = {int(m.metadata.src_node_id) for m in successful}
    errored_ids = {int(m.metadata.src_node_id) for m in errored}
    received_ids = successful_ids | errored_ids
    missing_ids = selected - received_ids

    client_rows = []
    for message in successful:
        metrics = message.content.get("metrics", MetricRecord())
        count = int(metrics.get(weighted_by_key, 0))
        row = {
            "node_id": int(message.metadata.src_node_id),
            "sample_count": count,
            "train_loss": float(metrics["train_loss"]) if "train_loss" in metrics else None,
        }
        client_rows.append(row)
    total_examples = sum(row["sample_count"] for row in client_rows)
    for row in client_rows:
        row["fedavg_sample_weight"] = (
            float(row["sample_count"] / total_examples) if total_examples else 0.0
        )

    failed_rows = [
        {
            "node_id": int(message.metadata.src_node_id),
            "reason": str(message.error.reason or "client returned an error"),
        }
        for message in errored
    ]
    failed_rows.extend({"node_id": node_id, "reason": "no reply received"} for node_id in sorted(missing_ids))
    weighted_losses = [
        (row["train_loss"], row["sample_count"])
        for row in client_rows
        if row["train_loss"] is not None and row["sample_count"] > 0
    ]
    train_loss = (
        sum(loss * count for loss, count in weighted_losses)
        / sum(count for _, count in weighted_losses)
        if weighted_losses else None
    )
    return {
        "round": int(server_round),
        "selected_client_count": len(selected),
        "successful_client_count": len(successful),
        "failed_client_count": len(failed_rows),
        "selected_node_ids": sorted(selected),
        "successful_clients": sorted(client_rows, key=lambda row: row["node_id"]),
        "failed_clients": sorted(failed_rows, key=lambda row: row["node_id"]),
        "aggregated_examples": int(total_examples),
        "aggregated_train_loss": train_loss,
    }


class AuditableFedAvg(FedAvg):
    """Flower FedAvg retaining its algorithm while recording round replies."""

    def __init__(self, *args, **kwargs):
        self.batchnorm_policy = kwargs.pop("batchnorm_policy", FEDAVG_FULL_STATE)
        if self.batchnorm_policy not in SUPPORTED_BATCHNORM_POLICIES:
            raise ValueError(f"unsupported BatchNorm aggregation policy: {self.batchnorm_policy}")
        super().__init__(*args, **kwargs)
        self.selected_nodes_by_round: dict[int, list[int]] = {}
        self.round_audit: dict[str, dict] = {}

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        messages = list(super().configure_train(server_round, arrays, config, grid))
        self.selected_nodes_by_round[server_round] = [
            int(message.metadata.dst_node_id) for message in messages
        ]
        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        messages = list(replies)
        self.round_audit[str(server_round)] = summarize_round_replies(
            server_round,
            self.selected_nodes_by_round.get(server_round, []),
            messages,
            weighted_by_key=self.weighted_by_key,
        )
        aggregated, metrics = super().aggregate_train(server_round, messages)
        if aggregated is not None and self.batchnorm_policy == FEDAVG_POOLED_BN:
            aggregated = _pool_batchnorm_buffers(aggregated, messages, self.weighted_by_key)
        return aggregated, metrics


def make_federated_strategy(name: str, **kwargs) -> AuditableFedAvg:
    """Select FedAvg or FedProx; both use sample-weighted FedAvg aggregation.

    FedProx differs in the client objective, where a proximal term is added
    during local optimization. The server aggregation remains FedAvg.
    """
    normalized = str(name).lower()
    if normalized not in SUPPORTED_FEDERATED_STRATEGIES:
        raise ValueError(
            f"unsupported federated strategy {name!r}; choose one of {SUPPORTED_FEDERATED_STRATEGIES}"
        )
    strategy = AuditableFedAvg(**kwargs)
    strategy.algorithm_name = normalized
    return strategy
