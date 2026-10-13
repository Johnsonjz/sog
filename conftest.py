"""Repo-root pytest bootstrap.

Makes ``import sog`` work from a fresh checkout (``src/`` layout) even when the
package has not been pip-installed.  When the package *is* installed (editable
or otherwise), this is a no-op — the installed copy wins if it is importable.
"""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
