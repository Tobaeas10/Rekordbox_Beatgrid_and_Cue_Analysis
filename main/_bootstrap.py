"""Puts the project root (the folder that contains lib/) on sys.path so every script can `import lib...`, and
re-runs the script with the project's .venv (created by SETUP) when it was started with another Python.
Each entry script starts with `import _bootstrap`. Identical copies live in main/ and dev/tools/ (only this one
switches to the .venv)."""
import os
import subprocess
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
ROOT = next(p for p in (_here, *_here.parents) if (p / 'lib').is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _switch_to_project_venv():
    venv_directory = ROOT / '.venv'
    venv_python = venv_directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not venv_python.is_file() or os.environ.get('BEATGRID_NO_VENV_SWITCH'):
        return
    if Path(sys.prefix).resolve() == venv_directory.resolve():
        return
    process = subprocess.Popen([str(venv_python), *sys.argv])
    while True:
        try:
            sys.exit(process.wait())
        except KeyboardInterrupt:
            continue                                # Ctrl+C reaches the child too; it decides how to stop


_switch_to_project_venv()
