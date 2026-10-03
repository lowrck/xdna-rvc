"""Unmodified-in-logic copies of the upstream RVC inference modules.

Source: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
        commit 81eed5e8f68b6bed1789f682fe78cdd324495afc (2026-08-04), infer/module/*.py, infer/rmvpe.py
License: MIT (see LICENSE in this directory).

Only import paths were changed. These modules are the *reference* implementation
used to validate the export models in xdna_rvc_tools.rvc.export_model; they are
not exported to ONNX themselves.
"""
