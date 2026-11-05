import json
import os

import numpy as np
import pytest
import torch
from flwr.common import ArrayRecord, ConfigRecord, Context, Error, Message, MetricRecord, RecordDict

from ecg_fl.data.partition import (
    assert_group_isolation,
    describe_partitions,
    partition_by_record,
)
from ecg_fl.evaluation.metrics import classification_metrics, evaluate_test_pair
from ecg_fl.evaluation.splits import grouped_stratified_folds
from ecg_fl.federated import client_app
from ecg_fl.federated.auditable_fedavg import (
    AuditableFedAvg, FEDAVG, FEDPROX, make_federated_strategy, summarize_round_replies,
)
from ecg_fl.federated.model_state import FEDAVG_FULL_STATE, model_state_for_fedavg
from ecg_fl.federated.model_state import FEDAVG_POOLED_BN
from ecg_fl.federated import simulation
from ecg_fl.models.cnn import ECGCNN
from ecg_fl.preprocessing.segmentation import AAMI_CLASSES
from ecg_fl.training import centralized
from ecg_fl.training.class_weights import compute_class_weights
from ecg_fl.training.engine import fedprox_penalty, train_model


def test_fedprox_penalty_matches_mu_over_two_squared_distance():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    reference = {"weight": torch.zeros_like(model.weight)}

    penalty = fedprox_penalty(model, reference, 0.2)
    penalty.backward()

    assert penalty.item() == pytest.approx(0.9)
    assert model.weight.grad.item() == pytest.approx(0.6)


def test_federated_strategy_selection_keeps_fedavg_default_and_accepts_fedprox():
    default_strategy = make_federated_strategy(FEDAVG, min_available_nodes=1)
    proximal_strategy = make_federated_strategy(FEDPROX, min_available_nodes=1)
    assert isinstance(default_strategy, AuditableFedAvg)
    assert default_strategy.algorithm_name == FEDAVG
    assert proximal_strategy.algorithm_name == FEDPROX
    with pytest.raises(ValueError, match="unsupported federated strategy"):
        make_federated_strategy("unknown")


def test_fedprox_client_passes_pretraining_global_snapshot(tmp_path, monkeypatch):
    x = np.ones((2, 252), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    np.savez_compressed(tmp_path / "partition_0.npz", x=x, y=y)
    monkeypatch.setenv("ECG_CLIENT_PARTITION_DIR", str(tmp_path))
    received = {}

    def fake_train(model, x, y, **kwargs):
        received.update(kwargs)
        received["snapshot"] = {
            name: value.clone() for name, value in kwargs["proximal_reference"].items()
        }
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        return [0.5]

    monkeypatch.setattr(client_app, "train_model", fake_train)
    initial = ECGCNN(input_length=252)
    before = {name: value.detach().clone() for name, value in initial.named_parameters()}
    request = Message(content=RecordDict({
        "arrays": ArrayRecord(initial.state_dict()),
        "config": ConfigRecord({
            "local-epochs": 1, "batch-size": 2, "learning-rate": 0.001,
            "weight-decay": 0.0001, "class-weighting": False, "seed": 42,
            "federated-strategy": FEDPROX, "proximal-mu": 0.4,
        }),
    }), dst_node_id=2, message_type="train")
    context = Context(run_id=1, node_id=2, node_config={"partition-id": 0}, state=RecordDict(), run_config={})

    client_app.train(request, context)

    assert received["proximal_mu"] == pytest.approx(0.4)
    assert all(torch.equal(received["snapshot"][name], value) for name, value in before.items())


def test_global_class_weights_are_training_only_canonical_and_finite():
    y_train = np.array([0] * 4 + [1] * 2, dtype=np.int64)
    weights, counts = compute_class_weights(y_train)
    assert counts.tolist() == [4, 2, 0, 0, 0]
    assert weights.tolist() == pytest.approx([0.3, 0.6, 0.0, 0.0, 0.0])
    assert np.isfinite(weights).all()
    with pytest.raises(ValueError):
        compute_class_weights(np.array([5]))


def test_central_and_flower_paths_receive_identical_global_weights(tmp_path, monkeypatch):
    global_weights = np.array([0.25, 2.0, 1.0, 4.0, 3.0], dtype=np.float32)
    seen_central = {}

    def fake_central_train(model, x, y, **kwargs):
        seen_central.update(kwargs)
        return [0.7]

    monkeypatch.setattr(centralized, "train_model", fake_central_train)
    centralized.run_centralized(
        np.ones((5, 252), dtype=np.float32), np.arange(5), epochs=1,
        class_weighting=True, class_weights=global_weights,
        output_path=tmp_path / "model.pt",
    )

    x_client = np.ones((4, 252), dtype=np.float32)
    y_client = np.array([0, 0, 1, 1], dtype=np.int64)  # classes 2–4 are absent locally
    np.savez_compressed(tmp_path / "partition_0.npz", x=x_client, y=y_client)
    monkeypatch.setenv("ECG_CLIENT_PARTITION_DIR", str(tmp_path))
    seen_client = {}

    def fake_client_train(model, x, y, **kwargs):
        seen_client.update(kwargs)
        return [0.8]

    monkeypatch.setattr(client_app, "train_model", fake_client_train)
    initial = ECGCNN(input_length=252)
    request = Message(content=RecordDict({
        "arrays": ArrayRecord(initial.state_dict()),
        "config": ConfigRecord({
            "local-epochs": 1, "batch-size": 4, "learning-rate": 0.001,
            "weight-decay": 0.0001, "class-weighting": True, "seed": 42,
            **{f"class-weight-{i}": float(v) for i, v in enumerate(global_weights)},
        }),
    }), dst_node_id=7, message_type="train")
    context = Context(run_id=1, node_id=7, node_config={"partition-id": 0}, state=RecordDict(), run_config={})
    reply = client_app.train(request, context)
    assert seen_central["class_weighting"] is True
    assert seen_client["class_weighting"] is True
    assert np.array_equal(seen_central["class_weights"], global_weights)
    assert np.array_equal(seen_client["class_weights"], global_weights)
    assert set(reply.content.keys()) == {"arrays", "metrics"}


def test_unweighted_mode_is_forwarded_and_absent_local_classes_are_safe(tmp_path, monkeypatch):
    seen_central = {}
    monkeypatch.setattr(
        centralized,
        "train_model",
        lambda model, x, y, **kwargs: (seen_central.update(kwargs) or [0.5]),
    )
    centralized.run_centralized(
        np.ones((4, 252), dtype=np.float32), np.array([0, 0, 1, 1]),
        epochs=1, class_weighting=False, class_weights=None,
        output_path=tmp_path / "unweighted.pt",
    )
    assert seen_central["class_weighting"] is False
    assert seen_central["class_weights"] is None

    x = np.random.default_rng(3).normal(size=(8, 252)).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    global_weights = np.array([0.25, 1.0, 3.0, 4.0, 2.0], dtype=np.float32)
    loss = train_model(
        ECGCNN(input_length=252), x, y, epochs=1, batch_size=4,
        class_weighting=True, class_weights=global_weights, device=torch.device("cpu"),
    )
    assert np.isfinite(loss).all()
    with pytest.raises(ValueError, match="global training split"):
        train_model(
            ECGCNN(input_length=252), x, y, epochs=1, batch_size=4,
            class_weighting=True, class_weights=None, device=torch.device("cpu"),
        )


def test_flower_class_weighting_false_disables_client_weights(tmp_path, monkeypatch):
    x = np.ones((4, 252), dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    np.savez_compressed(tmp_path / "partition_0.npz", x=x, y=y)
    monkeypatch.setenv("ECG_CLIENT_PARTITION_DIR", str(tmp_path))
    seen = {}
    monkeypatch.setattr(
        client_app,
        "train_model",
        lambda model, x, y, **kwargs: (seen.update(kwargs) or [0.25]),
    )
    model = ECGCNN(input_length=252)
    request = Message(content=RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "config": ConfigRecord({
            "local-epochs": 1, "batch-size": 4, "learning-rate": 0.001,
            "weight-decay": 0.0001, "class-weighting": False, "seed": 9,
            "batchnorm-policy": FEDAVG_FULL_STATE,
        }),
    }), dst_node_id=3, message_type="train")
    context = Context(run_id=1, node_id=3, node_config={"partition-id": 0}, state=RecordDict(), run_config={})
    client_app.train(request, context)
    assert seen["class_weighting"] is False
    assert seen["class_weights"] is None


def test_run_federated_serializes_class_weights_without_json_scope_error(tmp_path, monkeypatch):
    weights = np.array([0.2, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    def fake_run_simulation(**kwargs):
        model = ECGCNN(input_length=252)
        torch.save(model.state_dict(), tmp_path / "federated_model_state.pt")
        (tmp_path / "flower_history.json").write_text('{"rounds": 1}', encoding="utf-8")

    monkeypatch.setattr(simulation, "run_simulation", fake_run_simulation)
    monkeypatch.setenv("ECG_CLASS_WEIGHTS", "restore-after-test")
    x = np.ones((2, 252), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    model, history = simulation.run_federated(
        x, y, [np.array([0, 1])], rounds=1,
        output_dir=tmp_path, validation_data=(x[:1], y[:1]),
        class_weighting=True, class_weights=weights,
    )

    assert isinstance(model, ECGCNN)
    assert history == {"rounds": 1}
    assert json.loads(os.environ["ECG_CLASS_WEIGHTS"]) == pytest.approx(weights.tolist())


def test_subject_and_record_groups_never_cross_client_or_data_splits():
    records = np.repeat(["r1", "r2", "r3", "r4", "r5", "r6", "r7"], 5)
    subjects = np.repeat(["p12", "p12", "p3", "p4", "p5", "p6", "p7"], 5)
    labels = np.tile(np.arange(5), 7)
    parts = partition_by_record(labels, records, 5, alpha=0.2, seed=9, subjects=subjects)
    owners = {}
    for client_id, indices in enumerate(parts):
        for subject in set(subjects[indices]):
            assert subject not in owners or owners[subject] == client_id
            owners[subject] = client_id
    assert_group_isolation(records, subjects, {
        "train": np.arange(0, 10),
        "validation": np.arange(10, 20),
        "test": np.arange(20, 35),
    })
    with pytest.raises(ValueError, match="subjects leakage"):
        assert_group_isolation(records, subjects, {
            "train": np.arange(0, 5),
            "validation": np.arange(5, 10),
        })


def test_partition_summary_reports_counts_absent_labels_and_fedavg_weight():
    records = np.repeat(["a", "b", "c", "d", "e"], 10)
    subjects = records.copy()
    labels = np.concatenate([np.zeros(10, dtype=int), np.ones(10, dtype=int),
                             np.full(10, 2), np.full(10, 3), np.full(10, 4)])
    parts = partition_by_record(labels, records, 5, alpha=0, seed=2, subjects=subjects)
    rows = describe_partitions(labels, records, subjects, parts)
    assert sum(row["sample_count"] for row in rows) == len(labels)
    assert sum(row["fedavg_sample_weight"] for row in rows) == pytest.approx(1.0)
    assert all(row["records"] and row["subjects"] for row in rows)
    assert all(set(row["class_counts"]) == set(AAMI_CLASSES) for row in rows)


def test_grouped_stratified_folds_cover_development_once_without_subject_leakage():
    labels = np.tile(np.arange(5), 12)
    subjects = np.repeat([f"p{i}" for i in range(12)], 5)
    folds = grouped_stratified_folds(labels, subjects, n_splits=4, seed=13)
    seen = []
    for train, valid in folds:
        assert not (set(subjects[train]) & set(subjects[valid]))
        seen.extend(valid.tolist())
    assert sorted(seen) == list(range(len(labels)))


def test_fedavg_baseline_explicitly_includes_batchnorm_buffers():
    model = ECGCNN(input_length=252)
    state = model_state_for_fedavg(model, FEDAVG_FULL_STATE)
    assert "features.1.running_mean" in state
    assert "features.1.running_var" in state
    assert "features.1.num_batches_tracked" in state
    with pytest.raises(ValueError, match="unsupported BatchNorm"):
        model_state_for_fedavg(model, "unknown")


def test_auditable_fedavg_aggregates_full_model_state_by_sample_count():
    model_a = ECGCNN(input_length=252)
    model_b = ECGCNN(input_length=252)
    state_a = model_state_for_fedavg(model_a)
    state_b = model_state_for_fedavg(model_b)
    state_a["classifier.4.bias"].fill_(0.0)
    state_b["classifier.4.bias"].fill_(4.0)
    state_a["features.1.running_mean"].fill_(2.0)
    state_b["features.1.running_mean"].fill_(6.0)

    def reply(node_id, state, n, loss):
        request = Message(content=RecordDict(), dst_node_id=node_id, message_type="train")
        return Message(content=RecordDict({
            "arrays": ArrayRecord(state),
            "metrics": MetricRecord({"num-examples": n, "train_loss": loss}),
        }), reply_to=request)

    strategy = AuditableFedAvg(
        fraction_train=1.0, fraction_evaluate=0.0,
        min_train_nodes=2, min_evaluate_nodes=0, min_available_nodes=2,
    )
    strategy.selected_nodes_by_round[1] = [11, 22]
    aggregated, metrics = strategy.aggregate_train(
        1, [reply(11, state_a, 1, 0.2), reply(22, state_b, 3, 0.8)]
    )
    state = aggregated.to_torch_state_dict()
    assert state["classifier.4.bias"].numpy() == pytest.approx(np.full(5, 3.0))
    assert state["features.1.running_mean"].numpy() == pytest.approx(np.full(16, 5.0))
    assert strategy.round_audit["1"]["successful_client_count"] == 2
    assert strategy.round_audit["1"]["aggregated_examples"] == 4
    assert strategy.round_audit["1"]["aggregated_train_loss"] == pytest.approx(0.65)
    assert metrics["train_loss"] == pytest.approx(0.65)


def test_pooled_batchnorm_policy_combines_client_moments():
    states = []
    for mean_value, variance_value in ((2.0, 1.0), (6.0, 1.0)):
        model = ECGCNN(input_length=252)
        state = model_state_for_fedavg(model)
        state["features.1.running_mean"].fill_(mean_value)
        state["features.1.running_var"].fill_(variance_value)
        states.append(state)

    def reply(node_id, state, n):
        request = Message(content=RecordDict(), dst_node_id=node_id, message_type="train")
        return Message(content=RecordDict({
            "arrays": ArrayRecord(state),
            "metrics": MetricRecord({"num-examples": n, "train_loss": 0.1}),
        }), reply_to=request)

    strategy = AuditableFedAvg(
        fraction_train=1.0, fraction_evaluate=0.0,
        min_train_nodes=2, min_evaluate_nodes=0, min_available_nodes=2,
        batchnorm_policy=FEDAVG_POOLED_BN,
    )
    strategy.selected_nodes_by_round[1] = [11, 22]
    aggregated, _ = strategy.aggregate_train(1, [reply(11, states[0], 1), reply(22, states[1], 3)])
    result = aggregated.to_torch_state_dict()
    # Weighted means: 5; pooled population variance includes between-client spread.
    assert result["features.1.running_mean"].numpy() == pytest.approx(np.full(16, 5.0))
    assert result["features.1.running_var"].numpy() == pytest.approx(np.full(16, 4.0))


def test_round_audit_counts_successes_failures_missing_and_fedavg_weights():
    def request(node_id):
        return Message(content=RecordDict(), dst_node_id=node_id, message_type="train")

    reply_a = Message(content=RecordDict({"metrics": MetricRecord({"num-examples": 10, "train_loss": 0.5})}), reply_to=request(11))
    reply_b = Message(content=RecordDict({"metrics": MetricRecord({"num-examples": 20, "train_loss": 1.0})}), reply_to=request(22))
    failed = Message(error=Error(code=1, reason="synthetic failure"), reply_to=request(33))
    row = summarize_round_replies(2, [11, 22, 33, 44], [reply_a, reply_b, failed])
    assert row["selected_client_count"] == 4
    assert row["successful_client_count"] == 2
    assert row["failed_client_count"] == 2
    assert row["aggregated_examples"] == 30
    assert row["aggregated_train_loss"] == pytest.approx((0.5 * 10 + 1.0 * 20) / 30)
    assert [x["fedavg_sample_weight"] for x in row["successful_clients"]] == pytest.approx([1 / 3, 2 / 3])


def test_test_pair_uses_same_labels_and_zero_support_is_explicit():
    x = np.random.default_rng(7).normal(size=(5, 252)).astype(np.float32)
    y = np.arange(5, dtype=np.int64)
    reports = evaluate_test_pair(ECGCNN(), ECGCNN(), x, y)
    assert reports["centralized"]["per_class"]["Q"]["support"] == 1
    assert reports["federated"]["per_class"]["Q"]["support"] == 1
    sparse = classification_metrics(np.array([0, 0]), np.array([0, 0]))
    assert sparse["classes"] == list(AAMI_CLASSES)
    assert sparse["per_class"]["Q"]["support"] == 0
    assert sparse["per_class"]["Q"]["recall"] == 0
    assert np.isfinite(sparse["f1_macro"])
