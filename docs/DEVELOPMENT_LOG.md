# Development log

## 2026-11-04

- Inspected the four-page synopsis. Confirmed required MIT-BIH, 360 Hz signal pipeline, 0.5–45 Hz Butterworth, DWT, five AAMI classes, PyTorch 1D CNN and Flower FedAvg.
- Verified MIT-BIH dataset properties, current PhysioNet release, standard AAMI grouping, current SciPy filtering API, PyWavelets thresholding, and Flower 1.22 FedAvg/simulation references.
- Built the package modules, config, setup metadata, tests, and documentation scaffolding.
- Installed Python 3.9-compatible scientific dependencies, PyTorch 2.8.0, WFDB 4.2.0, Flower 1.22.0 and Ray 2.31.0.
- Implemented ingestion, DSP, annotation mapping/segmentation, record-safe non-IID allocation, CNN, local training, central baseline, Flower client/server simulation and metrics.
- Initial test run found Python 3.9 annotation incompatibilities; fixed. Final unit run: 5 passed.
- Initial archive and split figures were preliminary and are superseded by the unique-record rebuild documented below. The first archive accidentally duplicated record IDs; no experimental conclusions were taken from it.

## 2026-11-05

- Audited the evaluation split: 40 train records / 91,695 beats; records 201 and 202 from one subject are both held for validation (4,098 beats); six independent test records contain 13,659 beats. The test supports are N 5,832; S 110; V 1,116; F 376; Q 6,225.
- Corrected an early archive that contained duplicated record IDs from the first split draft. Rebuilt canonical DSP and no-DSP archives from the unique, disjoint 48-record list. Correct class totals per archive: N 90,594; S 2,781; V 7,235; F 802; Q 8,040 (109,452 beats).
- Changed record-level client partitioning to guarantee each of five clients has at least one whole training record. Seeded client train shuffling and Flower's global model initialization by the configured seed.
- Migrated final Flower execution to Flower 1.22's current `ClientApp`, `ServerApp`, `serverapp.strategy.FedAvg`, and `run_simulation` Simulation Runtime. Verified five nodes trained and Flower aggregated all five updates with zero failures. Global validation loss in a seeded smoke run fell from 1.5859 to 0.6513 after one round.
- Generated seeded paired real MIT-BIH smoke runs: one central epoch and one Flower round for DSP and no-DSP. Actual metrics, class sensitivities, client distributions, and caveats are in `docs/EXPERIMENTAL_RESULTS.md`; smoke performance is below project targets and Fusion sensitivity is 0%.
- Added direct Flower CLI preparation (`prepare-flower`), three notebooks, current API app metadata, preprocessing/class-distribution figures, confusion matrices, convergence curves, and paired DSP summary outputs.
- Ran the complete final test suite after adding dataset archive and Flower privacy-payload checks: 8 passed. CLI help for `ecg-fl` and `prepare-flower` also executed successfully.
- A full multi-epoch/multi-round study and repeated-seed stability analysis remain to be run; the executed runs are smoke experiments only.

## 2026-11-05 — Federated correction implementation review

- Read the complete 13-section implementation task before making changes. Preserved the configured held-out test records (102, 104, 107, 114, 118, 208) and the existing `outputs/demo_smoke/` artifacts. Did not install packages, run training, or use test records for model selection.
- Added a single global inverse-frequency class-weight computation from training labels and wired the same vector into centralized training and each Flower client; the configured weighting switch is honored and locally absent classes do not change client weights.
- Replaced the former Dirichlet row assignment with seeded subject/record-group assignment that prioritizes rare-class group spread where feasible and balances client sample loads. Added partition reports with record/subject membership, class counts/proportions, absent classes, and FedAvg sample weights.
- Expanded the fixed validation split to 101, 105, 201, 202, and 203 to supply support for all AAMI categories while preserving subject grouping and the untouched test list. Added optional development-only `StratifiedGroupKFold` central/federated cross-validation and paired test-evaluation helper; no CV or training run was launched.
- Made standard full-state sample-weighted FedAvg with BatchNorm buffers the explicit default. Added optional sample-weighted pooled BatchNorm running moments as an experimental policy, with a unit test. It is an approximation because BatchNorm running buffers are moving averages.
- Added actual per-round selected/success/failed/missing client accounting, sample counts and aggregation weights, weighted train loss, validation loss, and class support to Flower output. No ECG data are included in server metrics.
- Updated walkthrough, README, methodology, decisions, and result provenance notes. The smoke result file remains historical and explicitly is not evidence for the corrected implementation.
- Verification completed: the first full invocation exposed an omitted import in the new pooled-BatchNorm code (19 passed, 1 failed); the import was corrected. Final full suite: 20 passed, with five environment warnings (urllib3/LibreSSL and Matplotlib/PyParsing). CLI and grouped-CV help displayed successfully; `compileall` and `git diff --check` passed. The data-split audit confirmed 37/5/6 disjoint records and all 48 archive records assigned; test records and labels were not passed to training or CV. No full dataset training, test-set evaluation, or cross-validation model run was executed.
- Read-only split/partition audit on the prepared archive: train/validation/test contain 84,279 / 11,514 / 13,659 examples. Training N/S/V/F/Q counts are 74,153 / 2,483 / 5,417 / 422 / 1,804; the shared inverse-frequency weights are 0.2273 / 6.7885 / 3.1116 / 39.9427 / 9.3436. The five new client sample sizes are 18,274 / 17,596 / 14,989 / 18,170 / 15,250. Q remains concentrated: client 1 has 1,802, client 3 has 2, and the other three have none because Q support is concentrated in only two whole record groups. The 5-fold grouped split audit confirmed subject disjointness and no fixed-test records in any development fold; individual folds still have sparse or absent rare-class support, so aggregate OOF performance and fold supports must both be shown.
