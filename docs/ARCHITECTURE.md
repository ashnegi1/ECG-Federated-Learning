# Project architecture

1. `data/ingest.py` obtains the versioned MIT-BIH records and uses WFDB to read calibrated physical signals and `atr` beat annotations.
2. `preprocessing/dsp.py` applies a fourth-order zero-phase Butterworth bandpass at 0.5–45 Hz, then `db4` DWT soft-threshold denoising.
3. `preprocessing/segmentation.py` uses cardiologist-provided annotation sample positions as R-peak centers, cuts 0.25 s before and 0.45 s after, and maps original symbols into N/S/V/F/Q.
4. `data/partition.py` allocates complete records to clients, preventing beats from one record being divided among clinics.
5. `models/cnn.py` turns each (252 sample) beat into five logits using three Conv1D/BatchNorm/ReLU/MaxPool blocks and a small fully connected head.
6. `training/engine.py` supplies shared local and central training. The baseline and FL use the same model, training split, optimizer family, and independent test records.
7. `federated/client_app.py` and `server_app.py` implement Flower's current `ClientApp`/`ServerApp` interfaces. `federated/simulation.py` runs them with Flower's Simulation Runtime. Each client receives global parameters, performs local training on its local beat archive, and returns parameters plus sample count and local loss; Flower FedAvg aggregates those updates.
8. `evaluation/metrics.py` writes accuracy, macro precision/recall/F1, weighted F1, class report, confusion matrix, and figures.

## Privacy boundary

In this simulation, precomputed beats and labels are stored as separate ignored client shard files; each ClientApp loads only its assigned shard. Flower messages contain serialized model state arrays and scalar metadata; server aggregation never reads waveform tensors. Global validation data stays local to the ServerApp. This is data separation for a simulated experiment, not a cryptographic privacy guarantee. Ordinary FedAvg may leak information through model updates; secure aggregation and differential privacy are future work.

## Record split

Train records exclude validation and test records. Validation records are 101, 105, 201, 202, and 203; 201 and 202 are from the same subject and stay together. The fixed test records are 102, 104, 107, 114, 118, and 208. Client allocation is seeded and subject/record-grouped. It prioritizes rare-class group spread when supported by available record groups, and balances client sample loads within the configured slack. Shared training-only class weights are computed once for both the central model and all clients.

Flower uses full-state sample-weighted FedAvg by default, including BatchNorm buffers. Optional pooled BatchNorm moments are documented as a separate approximation. `AuditableFedAvg` records selected and replying clients, failures, client sample counts/weights, aggregated sample count/loss, and validation loss/support without raw signals.
