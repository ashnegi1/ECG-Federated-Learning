# Experimental methodology

## Dataset preparation

Download the configured records from PhysioNet MIT-BIH release 1.0.0. WFDB parses their signal and annotation formats. Use lead 0 (`MLII` when available) consistently. Filter each continuous record with a fourth-order Butterworth bandpass (0.5–45 Hz, forward-backward SOS filtering), then denoise with a level-4 `db4` DWT and universal soft threshold estimated from the finest-scale detail coefficients. Slice 0.25 s before and 0.45 s after each reference beat annotation (252 samples at 360 Hz).

## Evaluation split

Record membership is specified before training in YAML. Test and validation beats are not client training data. Validation records are 101, 105, 201, 202, and 203; records 201 and 202 represent the same subject and are both held in validation. The fixed test records are 102, 104, 107, 114, 118, and 208. Test metrics are calculated only after training; do not use them for tuning. Training-only inverse-frequency weights are calculated once and shared across centralized and federated optimization, while unweighted metrics and per-class support remain reported. Optional `cross-validate` uses subject-grouped folds over development records and excludes the fixed test set.

Client partitions are seeded, record/subject-grouped assignments. The allocator prioritizes spreading rare classes across groups where their record-level support permits, then targets similar client sample totals with a configurable slack fraction. Since entire record groups are indivisible, class balance is not guaranteed. FedAvg weighting remains proportional to each successful client's sample count.

The default BatchNorm policy is `fedavg_full_state`: standard Flower FedAvg averages model parameters and all BatchNorm buffers by client sample count. The optional `fedavg_pooled_bn` performs a sample-weighted moment pool of local running means/variances after parameter aggregation. BatchNorm running statistics are exponential moving averages rather than exact local population moments, so this alternative is an approximation and must be evaluated separately; standard full-state FedAvg remains the primary baseline.

## Centralized versus federated

Both use the same CNN and training records. The federated run assigns entire records to five clients using seeded Dirichlet client-size proportions, producing uneven client sizes and naturally varying label distributions across heterogeneous records. Each client runs local AdamW epochs; Flower's FedAvg combines state arrays weighted by local beat count. At test time, compare the resulting global model against the centralized model on the same fixed held-out records.

## DSP comparison

Prepare a second archive with `--no-dsp`. Keep records, segmentation, split, model seed, optimizer, client allocation, batch size, epochs and rounds fixed. Compare training losses and independent test metrics. Repeated seeds are recommended; a single run is descriptive, not inferential evidence of stability.

## Limitations

This dataset is small, old, class-imbalanced and not representative of modern clinical populations. Synthetic clinic partitions do not model network or device failure. Local simulation means clients share one host and is not deployment security. Performance targets in the synopsis are goals, not guaranteed findings.
