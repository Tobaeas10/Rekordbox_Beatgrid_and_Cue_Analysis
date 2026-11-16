"""Puts the project root (the folder that contains lib/) on sys.path so every script can `import lib...`.
Each entry script starts with `import _bootstrap`. Identical copies live in main/ and dev/tools/."""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
ROOT = next(p for p in (_here, *_here.parents) if (p / 'lib').is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
