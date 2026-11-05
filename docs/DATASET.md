# Dataset and labels

The project uses MIT-BIH Arrhythmia Database v1.0.0 from PhysioNet. It contains 48 two-channel, approximately half-hour records from 47 subjects, sampled at 360 Hz. WFDB reads `.hea`, `.dat` and reference `.atr` files. Downloaded raw records stay under ignored `data/raw/mitdb/`; processed beat arrays are also ignored.

## AAMI label map

| Group | MIT-BIH annotation symbols |
|---|---|
| N | N, L, R, e, j |
| S | A, a, J, S |
| V | V, E |
| F | F |
| Q | `/`, f, Q |

Unmapped rhythm, noise, and non-beat annotations are excluded, as are beats too close to a record boundary to extract the configured complete segment. Q includes paced beats `/`, fusion paced/normal `f`, and unclassifiable `Q` according to common MIT-BIH/AAMI mapping. Since Q is a heterogeneous remainder, report its per-class sensitivity and do not interpret it as a single clinical rhythm.

The synopsis requires all five groups. Accordingly, this implementation retains `f` as Q under the conventional map; some common evaluation protocols exclude paced records and do not score Q, so metrics from this project should not be compared directly to four-class benchmark scores.
