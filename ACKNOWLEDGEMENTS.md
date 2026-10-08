# Acknowledgements

CPSP is built on [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD).
We thank its authors, maintainers, and contributors for the IRSTD toolbox and
reference detector implementations. We also thank the authors of DNANet,
UIU-Net, ISNet, and [Torch-Pruning](https://github.com/VainF/Torch-Pruning).

This archive contains the **required source subset**, not the entire BasicIRSTD
toolbox: DNANet (1 file), UIUNet (2 files), ISNet and dependencies (8 files), and
the original `loss.py` / `utils.py` (2 files). All 13 upstream files retain their
original bytes and copyright notices. Paths and digests are recorded in
`upstream_source_sha256.json` and verified by `scripts/check.py`.

CPSP's data/metric runtime uses `core/data.py` and `core/metrics_ext.py`; unused
legacy `dataset.py` and `metrics.py` are not bundled. ISNet compatibility is
external in `compat/isnet.py`; the detector sources are not rewritten.

感谢 BasicIRSTD 及上述模型和 Torch-Pruning 的作者与贡献者。
上游源码保留各自许可，不被 CPSP 的 MIT 重新许可；详见 THIRD_PARTY_NOTICES.md。
