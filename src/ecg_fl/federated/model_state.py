"""Explicit model-state policies for BatchNorm under federated averaging."""
from __future__ import annotations

import torch


FEDAVG_FULL_STATE = "fedavg_full_state"
FEDAVG_POOLED_BN = "fedavg_pooled_bn"
SUPPORTED_BATCHNORM_POLICIES = (FEDAVG_FULL_STATE, FEDAVG_POOLED_BN)


def model_state_for_fedavg(model: torch.nn.Module, policy: str = FEDAVG_FULL_STATE) -> dict[str, torch.Tensor]:
    """Return the complete state sent by each client, including BN buffers.

    Both strategies need local running moments in client replies. The default
    subsequently averages every state entry as standard FedAvg does.
    """
    if policy not in SUPPORTED_BATCHNORM_POLICIES:
        raise ValueError(f"unsupported BatchNorm aggregation policy: {policy}")
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
