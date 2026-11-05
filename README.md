# Privacy-Preserving Federated Learning for ECG Arrhythmia Classification

An undergraduate ECE research implementation following `minor_project_synopsis_it2.pdf`. It uses MIT-BIH Arrhythmia Database beats, 0.5–45 Hz Butterworth filtering, DWT denoising, a lightweight PyTorch 1D CNN, record-partitioned simulated clinics, Flower FedAvg, and a centralized comparison.

## Quick start

Python 3.10 or 3.11 is recommended. Python 3.9 is supported by the pinned versions but is not recommended for a new environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m ecg_fl.cli prepare-data --download
python -m ecg_fl.cli run --epochs 2 --rounds 2
```

The defaults use five clients, one local epoch per round, and a 2-epoch central baseline if overridden as above. For a longer study, omit the short-run overrides and adjust `configs/default.yaml` after the smoke run. The first data preparation downloads 48 configured records (about 100 MB uncompressed) to ignored `data/raw/mitdb/` and writes processed beats to ignored `data/processed/mitdb.npz`.

## Pipeline and outputs

The entry point `ecg-fl` (or `python -m ecg_fl.cli`) provides these workflows:

- `prepare-data --download`: download official MIT-BIH records, read using WFDB, process continuous leads and segment labeled beats.
- `run`: train central and federated models on the configured training records and evaluate once on the independent test records.
- `cross-validate`: run optional subject-grouped cross-validation on development records, excluding the fixed test records.

For direct Flower CLI use, prepare local client archives and then launch the configured Flower App:

```bash
python -m ecg_fl.cli prepare-flower
flwr run . local-simulation --stream
```

The `run` command uses Flower's current ClientApp/ServerApp Simulation Runtime directly and saves both models, metric JSON and plots.

Experiment metrics and confusion plots are written under `outputs/experiment/`. The run reports raw values; no performance target is hard-coded. The archive is record tagged. Validation contains records 101, 105, 201, 202, and 203, with 201 and 202 kept together because they belong to the same subject. The fixed independent test records remain 102, 104, 107, 114, 118, and 208. Whole subject/record groups are assigned to clients; seeded allocation balances client sizes while retaining record-level label skew.

Training-only inverse-frequency class weights are computed once and shared with centralized training and all clients. Standard full-state FedAvg, including BatchNorm buffers, is the default. The optional `fedavg_pooled_bn` policy moment-pools local BatchNorm running statistics and should be treated as an experimental alternative, not canonical FedAvg. No corrected full experiment has yet been run after these implementation changes.

## Optional FedProx experiment

FedAvg remains the default. To run the same pipeline with FedProx, set `federated.strategy: fedprox` and choose `federated.proximal_mu` in `configs/default.yaml`, then run to a new output directory, for example:

```bash
python -m ecg_fl.cli run --output-dir outputs/fedprox
```

FedProx adds `mu / 2 * ||w - w_global||²` to each client's local loss, where `w_global` is a snapshot of the received trainable model parameters taken before local training. Server aggregation remains sample-weighted FedAvg; CNN architecture, data allocation, and BatchNorm handling are unchanged. Logged client training loss includes the proximal term in FedProx runs, so compare it with FedAvg loss accordingly. For direct Flower CLI runs, the equivalent settings are `strategy = "fedprox"` and `proximal-mu = 0.01` under `[tool.flwr.app.config]` in `pyproject.toml`.

## DSP comparison

```bash
python -m ecg_fl.cli prepare-data --no-dsp --output data/processed/mitdb_raw.npz
python -m ecg_fl.cli run --dataset data/processed/mitdb_raw.npz --output-dir outputs/no_dsp --epochs 2 --rounds 2
```

Use identical config, split, seeds, and training duration for the DSP and no-DSP runs. Compare the generated loss/metric reports; repeated seeds are needed to discuss stability.

## Google Colab

1. Upload or clone this project into a Colab runtime.
2. Set runtime type to T4 GPU if available (CPU remains supported).
3. Run `!pip install -r requirements.txt` and `!pip install -e .`.
4. Run `!python -m ecg_fl.cli prepare-data --download` and `!python -m ecg_fl.cli run`.
5. Download only the small `outputs/` results and model checkpoints if desired. Do not upload private ECG data; MIT-BIH is public.

The notebook workflow calls the package implementation; notebooks do not duplicate the model or preprocessing logic.

## Privacy statement

The project simulates client data separation in one process. Flower sends model parameters and small scalar metrics/counts for FedAvg. Raw waveforms and labels are not included in server messages or logs. FedAvg alone is not a cryptographic privacy guarantee and can leak information through model updates.

## Reproducibility and documentation

- YAML config and fixed random seeds: `configs/default.yaml`
- Detailed rationale and label mapping: `docs/DATASET.md`, `docs/DECISIONS_AND_DEVIATIONS.md`
- Architecture and experimental method: `docs/ARCHITECTURE.md`, `docs/METHODOLOGY.md`
- Chronological status: `docs/DEVELOPMENT_LOG.md`
- Notebooks: `notebooks/`
- Tests: `python -m pytest`

## Important limitations

The synopsis's >90% multi-class targets are not guaranteed. MIT-BIH is highly imbalanced and Q combines heterogeneous paced/unclassifiable beats. A single record split is not equivalent to patient-population clinical validation. Read result tables by per-class support and sensitivity, not accuracy alone.
