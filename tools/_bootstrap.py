"""Makes `xdna_rvc_tools` importable when a tool script is run directly."""

import logging
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(levelname)s] %(message)s")
    # Silence chatty third-party loggers.
    for name in ("torch", "onnx", "onnxscript", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
