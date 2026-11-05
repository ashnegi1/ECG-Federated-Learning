import numpy as np
import torch
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, RecordDict

from ecg_fl.data import ingest
from ecg_fl.data.ingest import build_dataset, save_dataset, load_dataset
from ecg_fl.preprocessing.dsp import butterworth_bandpass, dwt_denoise
from ecg_fl.preprocessing.segmentation import AAMI_CLASSES, ANNOTATION_TO_AAMI, extract_beats
from ecg_fl.data.partition import partition_by_record
from ecg_fl.models.cnn import ECGCNN
from ecg_fl.federated.flower_client import get_parameters, set_parameters, ECGFlowerClient
from ecg_fl.federated.client_app import train as flower_train
from ecg_fl.evaluation.metrics import classification_metrics


def test_mapping_and_segmentation():
    assert AAMI_CLASSES == ("N", "S", "V", "F", "Q")
    assert ANNOTATION_TO_AAMI["L"] == "N"
    assert ANNOTATION_TO_AAMI["A"] == "S"
    assert ANNOTATION_TO_AAMI["E"] == "V"
    assert ANNOTATION_TO_AAMI["F"] == "F"
    assert ANNOTATION_TO_AAMI["/"] == "Q"
    signal = np.arange(1000, dtype=np.float32)
    beats, labels, samples = extract_beats(signal, np.array([200, 400, 5]), ["N", "V", "N"], 100, .2, .4)
    assert len(beats) == 2 and labels == [0, 2] and samples == [200, 400]
    assert beats[0].shape == (60,)


def test_dataset_ingestion_save_and_load(tmp_path, monkeypatch):
    fs = 360
    signal = np.sin(np.arange(2000) / 10).astype(np.float32)
    monkeypatch.setattr(ingest, "load_record", lambda *_: (signal, np.array([500, 900]), ["N", "V"], fs))
    ds = build_dataset(tmp_path, ["001"], dsp=False, before_seconds=.25, after_seconds=.45)
    assert ds.x.shape == (2, 252)
    assert ds.y.tolist() == [0, 2]
    assert ds.records.tolist() == ["001", "001"]
    path = tmp_path / "beats.npz"
    save_dataset(ds, path)
    restored = load_dataset(path)
    assert np.array_equal(ds.x, restored.x) and np.array_equal(ds.y, restored.y)
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {"x", "y", "records", "subjects"}


def test_flower_client_reply_contains_only_model_and_scalar_metrics(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(10, 252)).astype(np.float32)
    y = np.tile(np.arange(5), 2)
    np.savez_compressed(tmp_path / "partition_0.npz", x=x, y=y)
    monkeypatch.setenv("ECG_CLIENT_PARTITION_DIR", str(tmp_path))
    model = ECGCNN(input_length=252)
    request = Message(content=RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "config": ConfigRecord({"local-epochs": 1, "batch-size": 10, "learning-rate": .001,
                                "weight-decay": .0001, "class-weighting": False, "seed": 5}),
    }), dst_node_id=7, message_type="train")
    context = Context(run_id=1, node_id=7, node_config={"partition-id": 0}, state=RecordDict(), run_config={})
    reply = flower_train(request, context)
    assert set(reply.content.keys()) == {"arrays", "metrics"}
    assert reply.content["metrics"]["num-examples"] == 10
    assert "train_loss" in reply.content["metrics"]
    assert all(isinstance(value, np.ndarray) for value in reply.content["arrays"].to_numpy_ndarrays())


def test_dsp_output_shape_and_frequency():
    fs = 360
    t = np.arange(3600) / fs
    raw = np.sin(2*np.pi*0.2*t) + np.sin(2*np.pi*10*t) + .3*np.sin(2*np.pi*70*t)
    filtered = butterworth_bandpass(raw, fs)
    denoised = dwt_denoise(filtered)
    assert filtered.shape == raw.shape and denoised.shape == raw.shape
    assert np.isfinite(denoised).all()
    assert np.std(filtered) < np.std(raw)


def test_partition_record_safe_and_reproducible():
    records = np.repeat(["r1", "r2", "r3", "r4", "r5", "r6"], 10)
    labels = np.tile(np.arange(5).repeat(2), 6)
    p1 = partition_by_record(labels, records, 3, alpha=.4, seed=3)
    p2 = partition_by_record(labels, records, 3, alpha=.4, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    owners = [{records[i] for i in part} for part in p1]
    assert sum(map(len, owners)) == len(set.union(*owners))


def test_cnn_forward_backward_and_serialization():
    model = ECGCNN(input_length=252)
    x = torch.randn(4, 252)
    logits = model(x)
    assert logits.shape == (4, 5)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    values = get_parameters(model)
    clone = ECGCNN(input_length=252)
    set_parameters(clone, values)
    assert all(np.allclose(a, b) for a, b in zip(values, get_parameters(clone)))


def test_local_client_training_and_metrics():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 252)).astype(np.float32)
    y = np.tile(np.arange(5), 4)
    client = ECGFlowerClient(x, y, {"local_epochs": 1, "batch_size": 10})
    initial = client.get_parameters({})
    updated, n, metrics = client.fit(initial, {"local_epochs": 1})
    assert n == 20 and "train_loss" in metrics
    assert any(not np.array_equal(a, b) for a, b in zip(initial, updated))
    report = classification_metrics(y, y)
    assert report["accuracy"] == 1.0
    assert np.asarray(report["confusion_matrix"]).shape == (5, 5)


def test_training_validation_curve_lengths():
    from ecg_fl.training.engine import train_model
    model = ECGCNN(input_length=252)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(15, 252)).astype(np.float32)
    y = np.tile(np.arange(5), 3)
    train_losses, val_losses = train_model(
        model, x, y, epochs=2, batch_size=5,
        validation_data=(x[:5], y[:5]), return_validation=True,
    )
    assert len(train_losses) == len(val_losses) == 2
    assert np.isfinite(train_losses + val_losses).all()
