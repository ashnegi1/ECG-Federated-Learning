# Technical references

1. PhysioNet, [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/): official records, annotations, download files and sampling details.
2. G. B. Moody and R. G. Mark, [The impact of the MIT-BIH Arrhythmia Database](https://doi.org/10.1109/51.932724), IEEE Engineering in Medicine and Biology Magazine, 2001.
3. P. de Chazal, M. O'Dwyer and R. B. Reilly, [Automatic classification of heartbeats using ECG morphology and heartbeat interval features](https://doi.org/10.1109/TBME.2004.827359), IEEE Transactions on Biomedical Engineering, 2004. Describes five ANSI/AAMI EC57 beat groups.
4. SciPy, [Butterworth filter design](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) and [forward-backward SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
5. PyWavelets, [multilevel 1D DWT](https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html) and [thresholding functions](https://pywavelets.readthedocs.io/en/stable/ref/thresholding-functions.html).
6. Flower, [PyTorch Quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html), [Flower App configuration](https://flower.ai/docs/framework/1.22/en/how-to-configure-pyproject-toml.html), and [FedAvg strategy reference](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html).
7. B. McMahan et al., [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html), AISTATS, 2017.

The specific label mapping, filter order, DWT threshold estimator, segment window, record split and current Flower API choice are documented in `DECISIONS_AND_DEVIATIONS.md` and `DATASET.md`.
