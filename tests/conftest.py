"""Test bootstrap: make `mjlab_microduck` importable even when the editable
install's .pth file is skipped.

The package is installed editable via a `mjlab_microduck.pth` in site-packages
that appends `src/` to sys.path. On some macOS setups an external file watcher
(Desktop sync / "cleaner" tools) sets the UF_HIDDEN flag on .pth files, and
CPython's site.addpackage silently SKIPS hidden .pth files — the import then
breaks until `chflags nohidden` is re-applied. Prepending src/ here makes the
test suite immune to that environment quirk (a no-op on Linux/CI where the
flag does not exist).
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
