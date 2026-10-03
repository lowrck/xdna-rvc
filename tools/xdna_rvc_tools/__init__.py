"""Model conversion and validation tooling for xdna-rvc.

This package converts RVC voice models and their shared neural components
(HuBERT/ContentVec content encoder, RMVPE pitch estimator) into ONNX assets for
the native runtime, and validates every export numerically against PyTorch.
"""

CONVERSION_VERSION = "1.0.0"
