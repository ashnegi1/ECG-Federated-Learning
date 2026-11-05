# Experimental results

> **Provenance:** The smoke metrics below predate the federated pipeline corrections. They are historical baseline artifacts only; no corrected central/federated training experiment has been run. Do not attribute these values to the corrected implementation. The original `outputs/demo_smoke/` files were preserved.

## Status

These are smoke experiments, not the full experiment requested for final research conclusions. Each run used one centralized epoch, one local epoch, and one Flower communication round. The same five-client whole-record partition seed, training records, test records, model architecture, batch size, optimizer, and metric code were used for the paired DSP/no-DSP runs. Results come from Flower's current ClientApp/ServerApp Simulation Runtime.

The selected evaluation split contains 40 training records (91,695 beats), two validation records (4,098 beats), and six held-out test records (13,659 beats). Records 201 and 202, which belong to the same subject, are both in validation. Test records are 102, 104, 107, 114, 118, and 208.

## Global test metrics

| Preprocessing | Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Butterworth + DWT | Centralized | 36.88% | 34.12% | 38.40% | 26.52% | 38.90% |
| Butterworth + DWT | Flower FedAvg | 41.76% | 16.75% | 35.71% | 22.25% | 30.42% |
| No DSP | Centralized | 57.27% | 38.89% | 42.50% | 36.18% | 57.92% |
| No DSP | Flower FedAvg | 19.02% | 17.69% | 26.63% | 15.16% | 18.99% |

## Per-class sensitivity on the test records

| Class | Test support | DSP centralized | DSP FedAvg | No-DSP centralized | No-DSP FedAvg |
|---|---:|---:|---:|---:|---:|
| N | 5,832 | 38.5% | 79.5% | 68.3% | 26.1% |
| S | 110 | 28.2% | 3.6% | 2.7% | 11.8% |
| V | 1,116 | 98.7% | 95.4% | 97.3% | 95.3% |
| F | 376 | 0.0% | 0.0% | 0.0% | 0.0% |
| Q | 6,225 | 26.7% | 0.0% | 44.2% | 0.0% |

The test set remains imbalanced; no condition achieved useful Fusion sensitivity. High overall accuracy therefore does not imply performance across all five classes.

## Convergence and DSP observation

| Run | Central train loss (epoch 1) | Central validation loss (epoch 1) | FedAvg weighted local train loss (round 1) | Global validation loss before → after round 1 |
|---|---:|---:|---:|---:|
| Butterworth + DWT | 0.6972 | 0.4548 | 0.9420 | 1.5859 → 0.6513 |
| No DSP | 0.7639 | 0.5387 | 1.0134 | 1.5870 → 1.3034 |

For this single seeded smoke run, the DSP condition had lower training and validation losses and a larger one-round reduction in global validation loss. Its held-out test accuracy and macro F1 were also lower than the no-DSP condition. One epoch and one seed cannot establish stability or a reliable DSP benefit; repeat with full training duration and multiple seeds before drawing a conclusion.

## Centralized versus federated

The smoke-run macro-F1 gaps were 4.27 percentage points with DSP and 21.02 points without DSP. The synopsis target of a 2–3 point gap was not reached. These runs are intentionally too short for that target to be judged conclusively.

## Artifacts

- Seeded DSP smoke: `outputs/seeded_dsp_smoke/`
- Seeded no-DSP smoke: `outputs/seeded_no_dsp_smoke/`
- Paired metric summary: `outputs/dsp_comparison.json`
- Per-run confusion matrices and convergence curves: each run directory

## Observed client heterogeneity

The seeded DSP run produced five clients with 2,260; 2,604; 7,865; 76,758; and 2,208 training beats. Their per-class distributions vary substantially; for example, client 4 received 1,802 Q beats and no S or F beats, while client 3 received 69,578 N beats. This is a strong synthetic Non-IID partition, not an estimate of real clinic populations.

All values are generated from model predictions; none are hard-coded. See `configs/default.yaml` for records and training settings. Do not report these as final model performance.
