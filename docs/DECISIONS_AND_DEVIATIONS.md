# Technical decisions and deviations

## Decisions

- MIT-BIH release 1.0.0 is pinned. This release has a documented correction to record 102's annotation; use the current PhysioNet version.
- Reference beat annotation samples are used as R-peak centers; no second R-peak detector is run, avoiding misalignment between labels and detected peaks.
- ECG windows are 0.25 s pre-peak and 0.45 s post-peak (252 samples at 360 Hz), configurable in `configs/default.yaml`.
- Filtering uses stable second-order sections with forward-backward application. This avoids phase shift of annotated beat locations.
- DWT uses `db4`, level 4 where the segment length supports it, with median absolute deviation noise estimate and universal soft threshold.
- The client partition assigns complete records; no training record is split across clients.
- Records 201 and 202 are both validation records to avoid subject leakage; the two records belong to the same patient. Test record 208 was selected to provide more Fusion examples than the earlier candidate split.

## Federated pipeline corrections (2026-11-05)

- Centralized and client training now consume the same global class-weight vector computed only from the current training rows. If a training class is absent, its weight is zero; client-local class absence does not alter the shared vector. Disabling class weighting sends no weights to either path.
- Client allocation is seeded and keeps records/subjects intact. It spreads rare record groups where possible and balances sample totals with `partition_slack_fraction`; whole-record constraints can still leave client label distributions unequal. This replaces the prior Dirichlet sample allocation, which could produce extreme client sizes and missing classes. This is a documented partitioning-method correction within the required Non-IID simulation.
- The fixed test records remain unchanged. Validation expands to 101, 105, 201, 202, and 203 to obtain support for all five classes; the rare F/Q validation support remains small. An optional `cross-validate` command offers subject-grouped development-only folds and never includes test records.
- Default BatchNorm aggregation is standard full-state sample-weighted FedAvg, including running-stat buffers. An explicit `fedavg_pooled_bn` option pools client running moments as an approximate sensitivity analysis; it is not the canonical result and has not been experimentally compared here.
- Per-round metadata records selected, successful, failed/missing clients, each successful client's sample count and FedAvg weight, aggregated sample count/loss, and global validation loss/support. The server receives model arrays and scalar counts/loss only, not raw ECG arrays.
- These corrections have unit/integration-level tests, but no corrected real-data training experiment was run as part of this implementation review. Historical smoke results in `EXPERIMENTAL_RESULTS.md` describe the previous implementation and must not be interpreted as results from the corrected pipeline.
- Q mapping includes paced `/`, fusion paced/normal `f`, and unclassifiable `Q`, to satisfy the synopsis five-class requirement.
- Flower 1.22 uses its current `ClientApp`, `ServerApp`, `flwr.serverapp.strategy.FedAvg`, and Simulation Runtime APIs. `run_simulation` launches these current App components for local and Colab execution; the Flower CLI app metadata is also configured for `flwr run`.

## Operational notes

WFDB 4.2's downloader repeatedly stalled during file metadata range checks in this environment. The download routine therefore streams exact MIT-BIH 1.0.0 files from PhysioNet's public S3 mirror; WFDB remains the signal and annotation reader. This is an acquisition transport adjustment only; it does not change the data source or format.

## Deviations

No research-scope deviation from the synopsis. Full-scale multi-epoch and multi-round experiments are still not complete; smoke results are recorded in `docs/EXPERIMENTAL_RESULTS.md`.
