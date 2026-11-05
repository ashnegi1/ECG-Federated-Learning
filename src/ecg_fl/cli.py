"""Command-line workflows for dataset preparation and experiments."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from ecg_fl.config.settings import Settings
from ecg_fl.data.ingest import download_mitdb, build_dataset, save_dataset, load_dataset
from ecg_fl.data.partition import partition_by_record, describe_partitions, assert_group_isolation
from ecg_fl.evaluation.metrics import classification_metrics, evaluate_test_pair, save_metrics, plot_confusion
from ecg_fl.evaluation.plots import plot_training_curves, plot_preprocessing_example, plot_class_distribution
from ecg_fl.evaluation.splits import grouped_stratified_folds
from ecg_fl.preprocessing.dsp import preprocess_ecg
from ecg_fl.federated.simulation import run_federated
from ecg_fl.models.cnn import ECGCNN
from ecg_fl.training.centralized import run_centralized
from ecg_fl.training.engine import predict
from ecg_fl.training.class_weights import compute_class_weights


def prepare(args):
    settings = Settings.load(args.config)
    data = settings.section("data")
    root = Path(data["root"])
    record_ids = data["records"] or (data["train_records"] + data["validation_records"] + data["test_records"])
    if args.download:
        download_mitdb(root, [str(v).zfill(3) for v in record_ids])
    ds = build_dataset(
        root, [str(v).zfill(3) for v in record_ids], lead=data["lead"],
        before_seconds=data["before_seconds"], after_seconds=data["after_seconds"],
        dsp=not args.no_dsp, dsp_config=settings.section("preprocessing"),
    )
    save_dataset(ds, args.output or data["processed"])
    print(f"Saved {len(ds.y)} beats from {len(np.unique(ds.records))} records to {args.output or data['processed']}")
    print("Class counts:", dict(Counter(ds.y.tolist())))
    if ds.raw_records and not args.no_dsp:
        key = sorted(ds.raw_records)[0]
        raw_signal, _, fs = ds.raw_records[key]
        processed_signal = preprocess_ecg(raw_signal, fs, **settings.section("preprocessing"))
        plot_preprocessing_example(raw_signal, processed_signal, fs, "outputs/figures/preprocessing_example.png")
    plot_class_distribution(ds.y, "outputs/figures/class_distribution.png")


def experiment(args):
    s = Settings.load(args.config)
    d, t, f = s.section("data"), s.section("training"), s.section("federated")
    ds = load_dataset(args.dataset or d["processed"])
    train_records = set(str(x).zfill(3) for x in d["train_records"])
    val_records = set(str(x).zfill(3) for x in d["validation_records"])
    test_records = set(str(x).zfill(3) for x in d["test_records"])
    configured_records = train_records | val_records | test_records
    archive_records = set(map(str, np.unique(ds.records)))
    if archive_records != configured_records:
        raise RuntimeError(
            f"archive/config record mismatch; missing={sorted(configured_records - archive_records)}, "
            f"unassigned={sorted(archive_records - configured_records)}"
        )
    x_train = np.flatnonzero(np.isin(ds.records, list(train_records)))
    x_val = np.flatnonzero(np.isin(ds.records, list(val_records)))
    x_test = np.flatnonzero(np.isin(ds.records, list(test_records)))
    if not len(x_train) or not len(x_test):
        raise RuntimeError("Dataset is missing training or held-out test records; run prepare-data over configured records")
    record_splits = {"train": train_records, "validation": val_records, "test": test_records}
    subject_splits = {
        name: set(map(str, np.unique(ds.subjects[np.flatnonzero(np.isin(ds.records, list(records_set)))])))
        for name, records_set in record_splits.items()
    }
    assert_group_isolation(ds.records, ds.subjects, {"train": x_train, "validation": x_val, "test": x_test})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    class_weighting = bool(t.get("class_weighting", False))
    class_weights, train_class_counts = compute_class_weights(ds.y[x_train])
    effective_weights = class_weights if class_weighting else None
    validation_counts = np.bincount(ds.y[x_val], minlength=5)
    test_counts = np.bincount(ds.y[x_test], minlength=5)
    partition_seed = f["seed"]
    partition_slack = float(f.get("partition_slack_fraction", f.get("partition_alpha", 0.3)))
    records_train = ds.records[x_train]
    subjects_train = ds.subjects[x_train]
    parts_local = partition_by_record(
        ds.y[x_train], records_train, f["clients"], partition_slack,
        partition_seed, subjects=subjects_train,
    )
    partitions_metadata = describe_partitions(ds.y[x_train], records_train, subjects_train, parts_local)
    experiment_metadata = {
        "seed": int(s.values["seed"]),
        "effective_config": {
            **s.values,
            "training": {**t, "central_epochs": args.epochs or t["central_epochs"]},
            "federated": {**f, "rounds": args.rounds or f["rounds"], "partition_slack_fraction": partition_slack},
        },
        "run_overrides": {"epochs": args.epochs, "rounds": args.rounds},
        "dataset_archive": str(args.dataset or d["processed"]),
        "record_split": {key: sorted(value) for key, value in record_splits.items()},
        "subject_split": {key: sorted(value) for key, value in subject_splits.items()},
        "class_order": ["N", "S", "V", "F", "Q"],
        "training_class_counts": dict(zip(("N", "S", "V", "F", "Q"), map(int, train_class_counts))),
        "class_weighting": class_weighting,
        "effective_class_weights": None if effective_weights is None else dict(
            zip(("N", "S", "V", "F", "Q"), map(float, effective_weights))
        ),
        "validation_support": dict(zip(("N", "S", "V", "F", "Q"), map(int, validation_counts))),
        "test_support": dict(zip(("N", "S", "V", "F", "Q"), map(int, test_counts))),
        "partition_algorithm": "seeded_subject_group_lpt_with_rare_group_spread",
        "partition_seed": partition_seed,
        "partition_slack_fraction": partition_slack,
        "client_partitions": partitions_metadata,
        "batchnorm_policy": f.get("batchnorm_policy", "fedavg_full_state"),
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(experiment_metadata, indent=2), encoding="utf-8"
    )
    np.random.seed(s.values["seed"])
    torch.manual_seed(s.values["seed"])
    model_c, training_c = run_centralized(
        ds.x[x_train], ds.y[x_train],
        validation_data=(ds.x[x_val], ds.y[x_val]) if len(x_val) else None,
        epochs=args.epochs or t["central_epochs"], batch_size=t["batch_size"],
        learning_rate=t["learning_rate"], class_weighting=class_weighting,
        class_weights=effective_weights, output_path=output / "centralized_model.pt",
        seed=s.values["seed"],
    )
    model_f, history = run_federated(
        ds.x[x_train], ds.y[x_train], parts_local,
        rounds=args.rounds or f["rounds"], local_epochs=t["local_epochs"],
        batch_size=t["batch_size"], learning_rate=t["learning_rate"],
        output_dir=output / "federated", seed=s.values["seed"],
        validation_data=(ds.x[x_val], ds.y[x_val]) if len(x_val) else None,
        class_weighting=class_weighting, class_weights=effective_weights,
        batchnorm_policy=f.get("batchnorm_policy", "fedavg_full_state"),
        federated_strategy=f.get("strategy", "fedavg"),
        proximal_mu=float(f.get("proximal_mu", 0.01)),
        partition_metadata=partitions_metadata,
    )
    pair_metrics = evaluate_test_pair(model_c, model_f, ds.x[x_test], ds.y[x_test])
    metrics_c = pair_metrics["centralized"]
    metrics_c.update(training_c)
    metrics_c["test_records"] = sorted(test_records)
    metrics_c["effective_class_weights"] = experiment_metadata["effective_class_weights"]
    save_metrics(metrics_c, output / "centralized_metrics.json")
    plot_confusion(metrics_c, output / "centralized_confusion.png")
    metrics_f = pair_metrics["federated"]
    metrics_f["training_history"] = history
    metrics_f["test_records"] = sorted(test_records)
    metrics_f["effective_class_weights"] = experiment_metadata["effective_class_weights"]
    save_metrics(metrics_f, output / "federated_metrics.json")
    plot_confusion(metrics_f, output / "federated_confusion.png")
    plot_training_curves(metrics_c, metrics_f, output / "training_curves.png")
    comparison = {
        "test_records": sorted(test_records),
        "class_order": ["N", "S", "V", "F", "Q"],
        "test_support": experiment_metadata["test_support"],
        "centralized": {key: metrics_c[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")},
        "federated": {key: metrics_f[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")},
    }
    comparison["federated_minus_centralized"] = {
        key: float(comparison["federated"][key] - comparison["centralized"][key])
        for key in comparison["centralized"]
    }
    comparison["per_class"] = {
        label: {
            "centralized": {key: metrics_c["per_class"][label][key] for key in ("precision", "recall", "f1-score", "support")},
            "federated": {key: metrics_f["per_class"][label][key] for key in ("precision", "recall", "f1-score", "support")},
            "federated_minus_centralized_recall": float(metrics_f["per_class"][label]["recall"] - metrics_c["per_class"][label]["recall"]),
            "federated_minus_centralized_f1": float(metrics_f["per_class"][label]["f1-score"] - metrics_c["per_class"][label]["f1-score"]),
        }
        for label in metrics_c["classes"]
    }
    (output / "comparison_report.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print("Centralized test accuracy:", metrics_c["accuracy"])
    print("Federated test accuracy:", metrics_f["accuracy"])
    print("Held-out records:", sorted(test_records))
    print("Validation support N/S/V/F/Q:", validation_counts.tolist())


def prepare_flower_app(args):
    """Create local per-client shards for direct `flwr run` app execution."""
    s = Settings.load(args.config)
    d, f = s.section("data"), s.section("federated")
    ds = load_dataset(args.dataset or d["processed"])
    train_records = set(str(x).zfill(3) for x in d["train_records"])
    val_records = set(str(x).zfill(3) for x in d["validation_records"])
    test_records = set(str(x).zfill(3) for x in d["test_records"])
    train_idx = np.flatnonzero(np.isin(ds.records, list(train_records)))
    val_idx = np.flatnonzero(np.isin(ds.records, list(val_records)))
    test_idx = np.flatnonzero(np.isin(ds.records, list(test_records)))
    assert_group_isolation(ds.records, ds.subjects, {"train": train_idx, "validation": val_idx, "test": test_idx})
    partition_slack = float(f.get("partition_slack_fraction", f.get("partition_alpha", 0.3)))
    parts = partition_by_record(
        ds.y[train_idx], ds.records[train_idx], f["clients"], partition_slack, f["seed"],
        subjects=ds.subjects[train_idx],
    )
    client_dir = Path(args.client_dir)
    client_dir.mkdir(parents=True, exist_ok=True)
    for p in client_dir.glob("partition_*.npz"):
        p.unlink()
    for cid, indices in enumerate(parts):
        if not len(indices):
            raise RuntimeError(f"Configured client {cid} received no records")
        np.savez_compressed(client_dir / f"partition_{cid}.npz", x=ds.x[train_idx][indices], y=ds.y[train_idx][indices])
        print(f"client {cid}: {len(indices)} beats from {len(np.unique(ds.records[train_idx][indices]))} records")
    client_metadata = describe_partitions(
        ds.y[train_idx], ds.records[train_idx], ds.subjects[train_idx], parts
    )
    class_weighting = bool(s.section("training").get("class_weighting", False))
    class_weights, class_counts = compute_class_weights(ds.y[train_idx])
    metadata = {
        "class_order": ["N", "S", "V", "F", "Q"],
        "training_class_counts": dict(zip(("N", "S", "V", "F", "Q"), map(int, class_counts))),
        "class_weighting": class_weighting,
        "class_weights": list(map(float, class_weights)) if class_weighting else None,
        "batchnorm_policy": f.get("batchnorm_policy", "fedavg_full_state"),
        "federated_strategy": f.get("strategy", "fedavg"),
        "proximal_mu": float(f.get("proximal_mu", 0.01)),
        "validation_records": sorted(val_records),
        "validation_support": dict(zip(("N", "S", "V", "F", "Q"), map(int, np.bincount(ds.y[val_idx], minlength=5)))),
        "partition_slack_fraction": partition_slack,
        "client_partitions": client_metadata,
    }
    Path("data/flower_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (client_dir / "partition_metadata.json").write_text(
        json.dumps(client_metadata, indent=2), encoding="utf-8"
    )
    validation_path = Path(args.validation_archive)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(validation_path, x=ds.x[val_idx], y=ds.y[val_idx])
    print(f"Saved validation archive with {len(val_idx)} beats to {validation_path}")


def cross_validate(args):
    """Run optional grouped CV on development records only; never touch test rows."""
    s = Settings.load(args.config)
    d, t, f = s.section("data"), s.section("training"), s.section("federated")
    ds = load_dataset(args.dataset or d["processed"])
    train_records = {str(x).zfill(3) for x in d["train_records"]}
    validation_records = {str(x).zfill(3) for x in d["validation_records"]}
    test_records = {str(x).zfill(3) for x in d["test_records"]}
    development_records = train_records | validation_records
    archive_records = set(map(str, np.unique(ds.records)))
    configured_records = development_records | test_records
    if archive_records != configured_records:
        raise RuntimeError(
            f"archive/config record mismatch; missing={sorted(configured_records - archive_records)}, "
            f"unassigned={sorted(archive_records - configured_records)}"
        )
    if development_records & test_records:
        raise RuntimeError("Development cross-validation records overlap the untouched test records")
    dev_idx = np.flatnonzero(np.isin(ds.records, list(development_records)))
    folds = grouped_stratified_folds(
        ds.y[dev_idx], ds.subjects[dev_idx], n_splits=args.folds, seed=s.values["seed"]
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof_true = {"centralized": [], "federated": []}
    oof_pred = {"centralized": [], "federated": []}
    fold_rows = []
    class_names = ("N", "S", "V", "F", "Q")
    class_weighting = bool(t.get("class_weighting", False))

    for fold_id, (train_rel, valid_rel) in enumerate(folds, start=1):
        train_idx, valid_idx = dev_idx[train_rel], dev_idx[valid_rel]
        assert_group_isolation(ds.records, ds.subjects, {"train": train_idx, "validation": valid_idx})
        train_groups = set(map(str, ds.subjects[train_idx]))
        valid_groups = set(map(str, ds.subjects[valid_idx]))
        if train_groups & valid_groups:
            raise RuntimeError(f"Subject leakage in grouped fold {fold_id}")
        fold_dir = out / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        weights, counts = compute_class_weights(ds.y[train_idx])
        effective_weights = weights if class_weighting else None
        parts = partition_by_record(
            ds.y[train_idx], ds.records[train_idx], f["clients"],
            float(f.get("partition_slack_fraction", f.get("partition_alpha", 0.3))),
            f["seed"] + fold_id, subjects=ds.subjects[train_idx],
        )
        clients = describe_partitions(ds.y[train_idx], ds.records[train_idx], ds.subjects[train_idx], parts)
        fold_metadata = {
            "fold": fold_id,
            "train_records": sorted(set(map(str, ds.records[train_idx]))),
            "validation_records": sorted(set(map(str, ds.records[valid_idx]))),
            "train_subjects": sorted(train_groups),
            "validation_subjects": sorted(valid_groups),
            "class_order": list(class_names),
            "training_class_counts": dict(zip(class_names, map(int, counts))),
            "validation_support": dict(zip(class_names, map(int, np.bincount(ds.y[valid_idx], minlength=5)))),
            "class_weighting": class_weighting,
            "effective_class_weights": None if effective_weights is None else dict(zip(class_names, map(float, effective_weights))),
            "client_partitions": clients,
            "batchnorm_policy": f.get("batchnorm_policy", "fedavg_full_state"),
        }
        (fold_dir / "experiment_metadata.json").write_text(json.dumps(fold_metadata, indent=2), encoding="utf-8")

        model_c, train_history = run_centralized(
            ds.x[train_idx], ds.y[train_idx], validation_data=(ds.x[valid_idx], ds.y[valid_idx]),
            epochs=args.epochs or t["central_epochs"], batch_size=t["batch_size"],
            learning_rate=t["learning_rate"], class_weighting=class_weighting,
            class_weights=effective_weights, output_path=fold_dir / "centralized_model.pt",
            seed=s.values["seed"] + fold_id,
        )
        pred_c = predict(model_c, ds.x[valid_idx])
        metric_c = classification_metrics(ds.y[valid_idx], pred_c)
        metric_c.update(train_history)
        metric_c["validation_records"] = fold_metadata["validation_records"]
        save_metrics(metric_c, fold_dir / "centralized_metrics.json")
        plot_confusion(metric_c, fold_dir / "centralized_confusion.png")

        model_f, history_f = run_federated(
            ds.x[train_idx], ds.y[train_idx], parts,
            rounds=args.rounds or f["rounds"], local_epochs=t["local_epochs"],
            batch_size=t["batch_size"], learning_rate=t["learning_rate"],
            output_dir=fold_dir / "federated", seed=s.values["seed"] + fold_id,
            validation_data=(ds.x[valid_idx], ds.y[valid_idx]),
            class_weighting=class_weighting, class_weights=effective_weights,
            batchnorm_policy=f.get("batchnorm_policy", "fedavg_full_state"),
            federated_strategy=f.get("strategy", "fedavg"),
            proximal_mu=float(f.get("proximal_mu", 0.01)),
            partition_metadata=clients,
        )
        pred_f = predict(model_f, ds.x[valid_idx])
        metric_f = classification_metrics(ds.y[valid_idx], pred_f)
        metric_f["training_history"] = history_f
        metric_f["validation_records"] = fold_metadata["validation_records"]
        save_metrics(metric_f, fold_dir / "federated_metrics.json")
        plot_confusion(metric_f, fold_dir / "federated_confusion.png")
        plot_training_curves(train_history, metric_f, fold_dir / "training_curves.png")

        for model_name, pred in (("centralized", pred_c), ("federated", pred_f)):
            oof_true[model_name].append(ds.y[valid_idx].copy())
            oof_pred[model_name].append(pred)
        fold_rows.append({
            "fold": fold_id,
            "validation_records": fold_metadata["validation_records"],
            "validation_support": fold_metadata["validation_support"],
            "centralized": {key: metric_c[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")},
            "federated": {key: metric_f[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")},
        })

    aggregate = {}
    for model_name in ("centralized", "federated"):
        y_true = np.concatenate(oof_true[model_name])
        y_pred = np.concatenate(oof_pred[model_name])
        aggregate[model_name] = classification_metrics(y_true, y_pred)
    report = {
        "method": "StratifiedGroupKFold over configured development records; subjects kept whole",
        "fold_count": args.folds,
        "test_records_excluded": sorted(test_records),
        "development_records": sorted(development_records),
        "class_order": list(class_names),
        "folds": fold_rows,
        "out_of_fold_metrics": aggregate,
    }
    (out / "cross_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "out_of_fold_predictions.npz",
        y_true=np.concatenate(oof_true["centralized"]),
        centralized=np.concatenate(oof_pred["centralized"]),
        federated=np.concatenate(oof_pred["federated"]),
    )
    print(f"Saved grouped cross-validation report to {out / 'cross_validation_report.json'}")


def main():
    parser = argparse.ArgumentParser(prog="ecg-fl")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare-data")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--download", action="store_true")
    p.add_argument("--no-dsp", action="store_true", help="For controlled raw-vs-DSP experiment; still segments annotated beats")
    p.add_argument("--output")
    p.set_defaults(func=prepare)
    p = sub.add_parser("run")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset")
    p.add_argument("--output-dir", default="outputs/experiment")
    p.add_argument("--epochs", type=int)
    p.add_argument("--rounds", type=int)
    p.set_defaults(func=experiment)
    p = sub.add_parser("prepare-flower")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset")
    p.add_argument("--client-dir", default="data/flower_clients")
    p.add_argument("--validation-archive", default="data/flower_validation.npz")
    p.set_defaults(func=prepare_flower_app)
    p = sub.add_parser("cross-validate")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset")
    p.add_argument("--output-dir", default="outputs/grouped_cv")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int)
    p.add_argument("--rounds", type=int)
    p.set_defaults(func=cross_validate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
