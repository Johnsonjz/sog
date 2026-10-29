"""Make the shared helpers and fixtures importable by the v2 test files."""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent  # test/

for path in (_HERE, _ROOT / "fixtures"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
