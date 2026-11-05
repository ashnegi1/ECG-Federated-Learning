# Project guidance

- Treat `minor_project_synopsis_it2.pdf` as the controlling project specification. Keep the PDF unchanged.
- Use MIT-BIH and WFDB, the 0.5–45 Hz Butterworth filter, DWT, R-peak segmentation, five AAMI classes, PyTorch 1D CNN, Flower FedAvg and record-level separation.
- Never fit on validation/test records. Never move raw signal data into shared server logs or artifacts.
- Do not fabricate experimental results. Keep metrics tied to actual runs and disclose limitations.
- Use explicit, reviewable config values. Record scientific decisions in `docs/DECISIONS_AND_DEVIATIONS.md` and significant progress in `docs/DEVELOPMENT_LOG.md`.
- Run tests using `python -m pytest`; avoid modifying unrelated files.
