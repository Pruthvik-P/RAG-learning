"""
Shared helper for the example scripts.

Adds the `src/` folder to Python's import path so `import rag` works when you
run these files directly, e.g.:

    python examples/01_loading.py

(If you `pip install -e .` the project, this shim is harmless.)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# On Windows the console default (cp1252) cannot print characters like the
# em-dash used in the sample docs. Force UTF-8 so output never looks garbled.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
