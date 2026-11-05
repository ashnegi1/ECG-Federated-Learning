# Project walkthrough: run, inspect, and demonstrate

This guide uses the repository's existing implementation and real MIT-BIH v1.0.0 data. It does not use mock training results. The short run below is a real-data smoke experiment with reduced training duration; use the configured full run for a longer experiment and do not present smoke results as final performance.

Run commands from the repository root unless a step says otherwise. The project currently has no standalone command that only runs DSP or segmentation: `prepare-data` invokes those existing modules, and the visualization examples below import those same modules directly.

## 1. Set up on an Apple M1 Mac

Use a native arm64 terminal. Check it with:

```bash
uname -m
```

Expected: `arm64`. Python 3.11 is recommended; the pinned dependencies also support Python 3.9. If Python 3.11 is not installed and you use Homebrew:

```bash
brew install python@3.11
```

Create the environment and install this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python --version
python -c "import platform, torch; print('architecture:', platform.machine()); print('MPS available:', torch.backends.mps.is_available())"
```

If you already use the supported system Python 3.9, replace `python3.11` with `python3`. Keep the environment activated for every later command. `.venv/` is ignored by Git.

Pinned core versions are listed in `requirements.txt` (including PyTorch 2.8.0, WFDB 4.2.0, and Flower 1.22.0). Installation and the first dataset preparation need internet access.

## 2. Get and prepare MIT-BIH

Download the configured MIT-BIH records and create the preprocessed beat archive:

```bash
MPLBACKEND=Agg python -m ecg_fl.cli prepare-data --download
```

This downloads public MIT-BIH files into `data/raw/mitdb/`, reads signals and reference annotations with WFDB, applies the configured Butterworth 0.5–45 Hz filter and DWT denoising, segments annotation-centered beats, maps labels to N/S/V/F/Q, and writes `data/processed/mitdb.npz`.

Expected console output includes the saved beat and record counts and class counts. The archive contains `x` (beat windows), `y` (class indices), `records`, and `subjects`. The already prepared project archive has 109,452 beats across 48 records; a fresh run should produce comparable counts with these pinned records/configuration. The downloader and preprocessing are CPU tasks. The first run takes longer and needs roughly 100 MB for the raw records plus additional space for archives.

To re-read existing raw data without downloading again, omit `--download`:

```bash
MPLBACKEND=Agg python -m ecg_fl.cli prepare-data
```

## 3. Entry points by component

| Component | Existing entry point | What it does |
|---|---|---|
| Dataset ingestion | `python -m ecg_fl.cli prepare-data --download`; implementation: `src/ecg_fl/data/ingest.py` | Downloads records, calls WFDB, collects reference annotations, saves archive. |
| DSP preprocessing | `src/ecg_fl/preprocessing/dsp.py` (`butterworth_bandpass`, `dwt_denoise`, `preprocess_ecg`) | Applies configured filter and wavelet denoising. Called by dataset preparation. |
| Heartbeat segmentation | `src/ecg_fl/preprocessing/segmentation.py` (`extract_beats`) | Uses MIT-BIH annotation sample positions as beat centers, slices configured windows, maps AAMI labels. |
| CNN architecture | `src/ecg_fl/models/cnn.py` (`ECGCNN`) | Defines the shared 1D convolutional model. |
| Centralized CNN training | `python -m ecg_fl.cli run`; implementation: `src/ecg_fl/training/centralized.py` | Trains on the configured training records and saves model/metrics. |
| Federated client training | `src/ecg_fl/federated/client_app.py` (`@app.train`) | Loads one local client shard and trains locally. |
| FedAvg and global validation | `src/ecg_fl/federated/server_app.py` (`AuditableFedAvg`, `evaluate_global`) | Flower aggregates model arrays and evaluates global parameters on validation records; per-round reply accounting is saved. |
| Flower simulation orchestration | `src/ecg_fl/federated/simulation.py` (`run_federated`) | Writes simulated shards and invokes Flower's Simulation Runtime. |
| Client partition preparation for Flower CLI | `python -m ecg_fl.cli prepare-flower` | Writes per-client archives from training records and a separate validation archive. |
| Model evaluation | `src/ecg_fl/evaluation/metrics.py`; invoked by `run` | Computes accuracy, macro precision/recall/F1, weighted F1, per-class report, and confusion matrix. |

The simplest complete entry point is `python -m ecg_fl.cli run`: it runs centralized training, the Flower simulation, and held-out evaluation. The model is trained with PyTorch. Centralized training uses Apple MPS when PyTorch reports it available; otherwise it uses CPU. Flower client training is explicitly CPU-only in the current implementation.

## 4. Verify the record split before training

The fixed record lists are in `configs/default.yaml`. The current split is 37 training records, validation records 101, 105, 201, 202, and 203, and test records 102, 104, 107, 114, 118, and 208. Records 201 and 202 belong to the same subject and are kept together in validation. The selected validation records contain support for N/S/V/F/Q, although F and Q support is small; grouped cross-validation is provided for a broader development-set assessment.

The experiment code selects training, validation, and test row indices separately from the `records` array. Only training rows are passed to `partition_by_record` and saved as client shards. Validation rows are used by the server for global validation loss. Test rows are only passed to `classification_metrics` after training. Client shards contain `x` and `y` from training rows only.

You can independently check the configured record sets and archive membership:

```bash
python - <<'PY'
import yaml, numpy as np
from pathlib import Path

cfg = yaml.safe_load(Path('configs/default.yaml').read_text())['data']
train = {str(r).zfill(3) for r in cfg['train_records']}
validation = {str(r).zfill(3) for r in cfg['validation_records']}
test = {str(r).zfill(3) for r in cfg['test_records']}
assert not (train & validation or train & test or validation & test)
with np.load(cfg['processed'], allow_pickle=False) as z:
    records = set(map(str, z['records']))
    print('train records present:', sorted(records & train))
    print('validation records present:', sorted(records & validation))
    print('test records present:', sorted(records & test))
    assert not ((records & train) & (records & validation))
    assert not ((records & train) & (records & test))
print('configured train/validation/test record sets are disjoint')
PY
```

This validates record separation, not cryptographic privacy. Simulated clients share one machine; FedAvg by itself is not a formal privacy guarantee.

## 5. Run tests, then a real-data demonstration

Run the project tests:

```bash
python -m pytest -q
```

The tests use small synthetic fixtures for unit checks; they are not experimental results. The test count and warnings should be taken from the latest test execution rather than assumed from this document.

For a real-data smoke run, use a new output directory so existing outputs remain intact:

```bash
MPLBACKEND=Agg python -m ecg_fl.cli run --epochs 1 --rounds 1 --output-dir outputs/demo_smoke
```

`--epochs` controls the centralized baseline; federated local epochs and other settings come from `configs/default.yaml`. The command trains and evaluates on real configured MIT-BIH records. It prints centralized and federated held-out test accuracy and lists test record IDs. These short-run metrics are for execution inspection only. The current review phase does not start a new training run; follow this command when you are ready to execute it yourself.

For the configured longer run (centralized 10 epochs, five Flower rounds, one local epoch per round):

```bash
MPLBACKEND=Agg python -m ecg_fl.cli run --output-dir outputs/full_run
```

The longer run is computationally expensive. It runs on the M1, but the five Flower client workers are configured for CPU. Expect more time than for dataset preparation or tests; close unrelated memory-heavy applications. Reuse a new output directory for each experiment so results remain easy to compare.

Optional subject-grouped cross-validation trains separate models across folds, so it is substantially more expensive. It excludes all fixed test records:

```bash
MPLBACKEND=Agg python -m ecg_fl.cli cross-validate --folds 5 --output-dir outputs/grouped_cv
```

This command has not been run during the implementation review. Its report and fold outputs are generated only when you explicitly run it.

### Direct Flower CLI alternative

The main `run` command already performs Flower simulation. To inspect the Flower App entry point separately:

```bash
python -m ecg_fl.cli prepare-flower
flwr run . local-simulation --stream
```

This writes ignored `data/flower_clients/partition_*.npz` files and `data/flower_validation.npz`, then launches the configured Flower app. It uses real archive rows. The main `run` command writes its own shards under the selected run's `federated/client_data/` directory and does not require this separate preparation step.

The direct `flwr run` path is useful for inspecting Flower's App lifecycle and writes its global model/history under `outputs/federated/`; it evaluates validation loss only. Use the main `run` command when you need the independent held-out test metrics and confusion matrices.

## 6. Inspect and visualize signals and outputs

### Raw, filtered, and wavelet-denoised ECG

Dataset preparation saves a convenience raw-versus-final-preprocessed plot at `outputs/figures/preprocessing_example.png`. The following command shows raw, filtered-only, and filtered-plus-DWT traces separately, using the existing DSP implementation, and saves the figure:

```bash
MPLBACKEND=Agg python - <<'PY'
from pathlib import Path
import matplotlib.pyplot as plt
import wfdb
from ecg_fl.preprocessing.dsp import butterworth_bandpass, dwt_denoise

rec = wfdb.rdrecord('data/raw/mitdb/100', physical=True)
raw = rec.p_signal[:3600, 0]
filtered = butterworth_bandpass(raw, rec.fs)
denoised = dwt_denoise(filtered)
t = range(len(raw))
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, sig, title in zip(axes, (raw, filtered, denoised),
                          ('Raw MIT-BIH lead', 'Butterworth 0.5–45 Hz', 'Butterworth + DWT db4')):
    ax.plot([i / rec.fs for i in t], sig, linewidth=.8)
    ax.set_ylabel('mV'); ax.set_title(title); ax.grid(alpha=.2)
axes[-1].set_xlabel('Time (s)')
fig.tight_layout()
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
fig.savefig('outputs/figures/raw_filtered_denoised.png', dpi=160)
print('Saved outputs/figures/raw_filtered_denoised.png')
PY
```

Open the saved figure on macOS with `open outputs/figures/raw_filtered_denoised.png`.

### Segmented heartbeats

This plots one real processed beat from each AAMI class available in the archive, using existing segmented rows and label definitions:

```bash
MPLBACKEND=Agg python - <<'PY'
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ecg_fl.preprocessing.segmentation import AAMI_CLASSES

with np.load('data/processed/mitdb.npz', allow_pickle=False) as z:
    x, y = z['x'], z['y']
fig, axes = plt.subplots(len(AAMI_CLASSES), 1, figsize=(9, 10), sharex=True)
for class_id, (ax, label) in enumerate(zip(axes, AAMI_CLASSES)):
    indices = np.flatnonzero(y == class_id)
    if not len(indices):
        ax.set_title(f'{label}: no beats in this archive')
        continue
    ax.plot(np.arange(x.shape[1]) / 360 - .25, x[indices[0]])
    ax.set_title(f'{label}: example from record-centered beat archive')
    ax.set_ylabel('mV'); ax.grid(alpha=.2)
axes[-1].set_xlabel('Seconds relative to annotated R-peak')
fig.tight_layout()
Path('outputs/figures').mkdir(parents=True, exist_ok=True)
fig.savefig('outputs/figures/segmented_beats.png', dpi=160)
print('Saved outputs/figures/segmented_beats.png')
PY
```

Open it on macOS with `open outputs/figures/segmented_beats.png`.

### Loss curves and centralized-versus-federated performance

Each `run` automatically writes `training_curves.png` with actual centralized and federated train/validation losses. It also writes real metric JSON and confusion-matrix PNGs. To make a direct macro-F1 comparison from a completed run:

```bash
MPLBACKEND=Agg python - <<'PY'
import json
import matplotlib.pyplot as plt
from pathlib import Path

root = Path('outputs/demo_smoke')
names = ('centralized', 'federated')
metrics = [json.loads((root / f'{name}_metrics.json').read_text()) for name in names]
values = [m['f1_macro'] for m in metrics]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(names, values, color=('#2878b5', '#69b3e7'))
ax.bar_label(bars, fmt='%.3f')
ax.set(ylabel='Macro F1', title='Held-out MIT-BIH test performance', ylim=(0, 1))
ax.grid(axis='y', alpha=.25); fig.tight_layout()
fig.savefig(root / 'performance_comparison.png', dpi=160)
print('Saved', root / 'performance_comparison.png')
PY
```

Open the curves, performance chart, and confusion matrices on macOS with:

```bash
open outputs/demo_smoke/training_curves.png
open outputs/demo_smoke/performance_comparison.png
open outputs/demo_smoke/centralized_confusion.png
open outputs/demo_smoke/federated_confusion.png
```

The run also creates `centralized_confusion.png` and `federated_confusion.png`. To print per-class precision, recall/sensitivity, F1, and support from the actual results:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('outputs/demo_smoke')
for name in ('centralized', 'federated'):
    metrics = json.loads((root / f'{name}_metrics.json').read_text())
    print('\n', name, 'accuracy=', metrics['accuracy'], 'macro F1=', metrics['f1_macro'])
    for label in metrics['classes']:
        row = metrics['per_class'][label]
        print(label, 'precision=', row['precision'], 'sensitivity/recall=', row['recall'],
              'F1=', row['f1-score'], 'support=', int(row['support']))
PY
```

Interpret class metrics together with support. In particular, an overall accuracy can hide poor sensitivity on rare classes.

## 7. DSP versus no-DSP controlled run

Prepare a second archive with filtering and denoising disabled (the annotations, segmentation, and records stay the same):

```bash
MPLBACKEND=Agg python -m ecg_fl.cli prepare-data --no-dsp --output data/processed/mitdb_raw.npz
MPLBACKEND=Agg python -m ecg_fl.cli run --dataset data/processed/mitdb_raw.npz --epochs 1 --rounds 1 --output-dir outputs/demo_no_dsp
```

Use the same seed, training duration, client count, and records in both conditions. The code's `compare_dsp_runs` helper can create a paired chart from completed metric JSON:

```bash
MPLBACKEND=Agg python - <<'PY'
from ecg_fl.evaluation.plots import compare_dsp_runs
compare_dsp_runs('outputs/demo_smoke', 'outputs/demo_no_dsp', 'outputs/dsp_comparison.json')
print('Saved outputs/dsp_comparison.json and outputs/dsp_comparison.png')
PY
```

One seed and one epoch/round are descriptive only. Repeat with longer training and multiple seeds before making a claim about convergence stability or DSP benefit.

## 8. Where files are written

| Path | Contents |
|---|---|
| `data/raw/mitdb/` | Downloaded public `.hea`, `.dat`, `.atr` records. |
| `data/processed/mitdb.npz` | Processed beat windows, labels, record IDs, subject IDs. |
| `outputs/figures/` | Dataset-preparation raw/final-preprocessed and class-distribution plots; walkthrough signal/beat plots. |
| `outputs/<run-name>/centralized_model.pt` | Centralized model state. |
| `outputs/<run-name>/federated/` | Flower model state, history, local simulated client shards, validation archive. |
| `outputs/<run-name>/*_metrics.json` | Actual test metrics, class report, confusion matrix, training history where applicable. |
| `outputs/<run-name>/*_confusion.png` | Actual held-out test confusion matrix plots. |
| `outputs/<run-name>/training_curves.png` | Central and federated loss curves. |
| Terminal | CLI progress, test output, Flower/Ray logs; the current code does not create a general persistent text log. |

`data/`, `outputs/`, model checkpoints, and generated arrays/plots are Git-ignored. Do not add them to staging. Recreate them by running the documented commands.

## 9. Mac versus Google Colab

- **Apple M1 Mac:** suitable for setup, unit tests, data preparation, plots, short real-data runs, and longer CPU/MPS experiments if runtime is acceptable. Centralized PyTorch selects MPS when available. Flower clients explicitly train on CPU.
- **Google Colab:** can run the same CPU project code and is useful if its memory or session environment is more convenient. However, the present code does **not** select CUDA: its general device selector chooses MPS or CPU, and the Flower simulation allocates zero GPUs and pins client training to CPU. A Colab T4 GPU therefore will not accelerate these commands without source changes. No source change is made in this walkthrough phase.
- **Compute-heavy steps:** full archive preparation is CPU/I/O work; the 10-epoch central run and five-round/five-client Flower run are the costly parts. Tests, CLI inspection, and plotting prepared results are lightweight.

To run the same commands in Colab, first upload the project folder to the runtime (there is no configured remote URL to clone), then in a notebook cell:

```python
%cd /content/ECG-Federated-Learning
!python -m pip install -r requirements.txt
!python -m pip install -e .
!python -m ecg_fl.cli prepare-data --download
!python -m ecg_fl.cli run --epochs 1 --rounds 1 --output-dir outputs/demo_smoke
```

For full training, replace the last command with `!python -m ecg_fl.cli run --output-dir outputs/full_run`. Colab storage is temporary; copy out only the desired small results after the run. The current code remains CPU-bound on Colab despite selecting a GPU runtime, as explained above.

The notebooks are optional inspection aids, not separate implementations. If using Jupyter locally, install JupyterLab in the active environment and start it from the notebook directory so the notebooks' relative `../data` paths resolve:

```bash
python -m pip install jupyterlab
cd notebooks
jupyter lab
```

## 10. Recommended demonstration sequence

1. Show the project modules and `configs/default.yaml`; explain that record membership is fixed before training.
2. Show the data and label mapping in `docs/DATASET.md`; run the split check above.
3. Run the tests and explain that synthetic fixtures validate code behavior but are not study results.
4. Run `prepare-data --download` if the local archive is not already present. Inspect class counts and the raw/processed figures.
5. Generate and explain the three-stage signal plot and representative segmented-beat plot.
6. Run the one-epoch/one-round real-data smoke experiment to show centralized learning, client-local updates, FedAvg, and held-out evaluation.
7. Inspect loss curves, both confusion matrices, and per-class metrics; discuss class support and any weak classes honestly.
8. Run the no-DSP control with identical settings, compare outputs, and state that repeated longer runs are required for scientific conclusions.
9. If time permits, launch a new full-run output directory and describe it as in progress until it actually finishes.

## Troubleshooting

- **`ModuleNotFoundError: ecg_fl`:** activate `.venv` and run `python -m pip install -e .` from the repository root.
- **Python/architecture mismatch:** check `python --version` and `python -c "import platform; print(platform.machine())"`; use a native arm64 Python environment and the pinned dependencies.
- **Dataset not found:** run `python -m ecg_fl.cli prepare-data --download` from the repository root; check internet access and that `data/raw/mitdb/` is writable.
- **Slow first run:** full record parsing, DSP, and beat extraction happen before training. Use a short real-data smoke run for a demonstration; do not replace it with hard-coded sample metrics.
- **Flower/Ray startup or resource failure:** close other heavy workloads, confirm all packages installed from `requirements.txt`, and try a single-round run first. The simulation uses five CPU client workers by default.
- **Apple MPS error:** check the printed MPS availability. Centralized training falls back to CPU when MPS is unavailable; federated client workers already use CPU.
- **Missing output plot:** confirm the command completed and inspect its selected `--output-dir`. Plots are written after the associated training/evaluation finishes.
- **Matplotlib aborts while creating a figure in a headless terminal:** prefix the command with `MPLBACKEND=Agg`; the walkthrough uses this for scripts that save PNGs, then opens the saved files separately.
- **Notebook data path errors:** start Jupyter from `notebooks/` as described above, or execute the root-relative command examples from the repository root.

## Current validation and limits

The current configured archive contains 48 records split into 37 training, five validation, and six test records. Tests and record-isolation checks were run during implementation review; the completion report records the exact latest outcome. No new model experiment was run as part of that review.

The previously recorded one-epoch/one-round outputs in `docs/EXPERIMENTAL_RESULTS.md` are actual smoke-run outputs, not guaranteed expected results. Your new run can differ slightly by hardware/runtime. The project has not yet completed a full multi-round, multi-seed study; do not claim the synopsis's target scores without measuring them.
