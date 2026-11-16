#!/usr/bin/env python3
"""
populate_env.py - find the Rekordbox paths/key the analysis scripts need and write them into .env.

Detects:
  REKORDBOX_DATABASE_PATH   path of the live master.db   (pyrekordbox config, then well-known locations)
  REKORDBOX_ENCRYPTION_KEY  SQLCipher key                (existing .env value if it works, else captured from
                                        pyrekordbox's own DEBUG log / its built-in blob)
Also checks ffmpeg / ffprobe on PATH and the share folder next to master.db.

.env handling: values you already set and that still work are KEPT. Empty/missing/broken ones
are filled. Comments and unknown lines are preserved. Missing optional settings get the defaults.
If no .env exists, `_env` (the template) next to this script is used as the starting point.
Every path that could not be found is listed at the end so you know what to fill in by hand.

Run:  python populate_env.py            (add --force to overwrite working values too,
                                              --no-search to skip the disk search for qaac)
"""
import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from lib import paths

ENV_PATH = paths.ENV_FILE
TEMPLATE_PATH = paths.DATA / "_env"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---- pyrekordbox imports (same layout fallbacks as lib/rekordbox/database.py) --------------------
try:
    from pyrekordbox.masterdb import MasterDatabase as RekordboxDatabase
except ImportError:
    try:
        from pyrekordbox import Rekordbox6Database as RekordboxDatabase
    except ImportError:
        RekordboxDatabase = None

DEFAULTS = []
REQUIRED = ("REKORDBOX_DATABASE_PATH", "REKORDBOX_ENCRYPTION_KEY")


# ---------------------------------------------------------------------------------------
# .env read / write
# ---------------------------------------------------------------------------------------
def parse_env(text):
    settings = {}
    for line in text.splitlines():
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith("#") and "=" in stripped_line:
            name, value = stripped_line.split("=", 1)
            settings[name.strip()] = value.strip().strip('"').strip("'")
    return settings


def set_value(lines, key, value):
    """Replace KEY=... in place, or append. Returns the new list of lines."""
    line_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for i, line in enumerate(lines):
        if line_pattern.match(line):
            lines[i] = f"{key}={value}"
            return lines
    lines.append(f"{key}={value}")
    return lines


# ---------------------------------------------------------------------------------------
# REKORDBOX_DATABASE_PATH
# ---------------------------------------------------------------------------------------
def pyrekordbox_database_path():
    try:
        from pyrekordbox.config import get_config
    except ImportError:
        return None, "pyrekordbox not installed"
    for section in ("rekordbox7", "rekordbox6"):      # same preference order pyrekordbox uses
        try:
            configured_path = get_config(section).get("db_path")
        except Exception:
            configured_path = None
        if configured_path and Path(configured_path).exists():
            return Path(configured_path), f"pyrekordbox config ({section})"
    return None, "pyrekordbox found no Rekordbox 6/7 install"


def known_database_locations():
    locations = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        locations += [Path(appdata) / "Pioneer" / folder_name / "master.db" for folder_name in ("rekordbox", "rekordbox6")]
    home = Path.home()
    locations += [home / "Library" / "Pioneer" / folder_name / "master.db" for folder_name in ("rekordbox", "rekordbox6")]
    locations += [home / "AppData" / "Roaming" / "Pioneer" / folder_name / "master.db" for folder_name in ("rekordbox", "rekordbox6")]
    return locations


def find_database(existing, force):
    if existing and not force and Path(existing).is_file():
        return Path(existing), "kept from .env"
    configured_path, source_description = pyrekordbox_database_path()
    if configured_path:
        return configured_path, source_description
    for candidate in known_database_locations():
        if candidate.is_file():
            return candidate, "well-known location"
    return None, source_description


# ---------------------------------------------------------------------------------------
# REKORDBOX_ENCRYPTION_KEY
# ---------------------------------------------------------------------------------------
class _KeyCatcher(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.key = None

    def emit(self, record):
        match = re.match(r"^Key:\s*(\S+)", record.getMessage())
        if match:
            self.key = match.group(1)


def try_open(database_path, key=""):
    """Open the DB the way the other scripts do. Returns (ok, error)."""
    if RekordboxDatabase is None:
        return False, "pyrekordbox not installed"
    keyword_arguments = {"path": str(database_path)}
    if key:
        keyword_arguments["key"] = key
    try:
        database = RekordboxDatabase(**keyword_arguments)
    except Exception as e:                                  # wrong key, missing sqlcipher, ...
        return False, f"{type(e).__name__}: {e}"
    try:
        database.close()
    except Exception:
        pass
    return True, ""


def capture_key(database_path):
    """Open without a key and read the key pyrekordbox logs at DEBUG level."""
    pyrekordbox_logger = logging.getLogger("pyrekordbox")
    old_level = pyrekordbox_logger.level
    key_catcher = _KeyCatcher()
    pyrekordbox_logger.addHandler(key_catcher)
    pyrekordbox_logger.setLevel(logging.DEBUG)
    try:
        opened, error = try_open(database_path)
    finally:
        pyrekordbox_logger.removeHandler(key_catcher)
        pyrekordbox_logger.setLevel(old_level)
    return (key_catcher.key if opened else None), error, key_catcher.key


def builtin_key():
    """Fallback: pyrekordbox's own obfuscated key blob (pyrekordbox >= 0.4/0.5 layout)."""
    try:
        from pyrekordbox.utils import deobfuscate
        try:
            from pyrekordbox.masterdb.database import BLOB
        except ImportError:
            from pyrekordbox.db6.database import BLOB
        return deobfuscate(BLOB)
    except Exception:
        return None


def find_key(database_path, existing, force):
    """Returns (key or None, note)."""
    if existing and not force:
        opened, _ = try_open(database_path, existing)
        if opened:
            return existing, "kept from .env (verified: database opens with it)"
    key, error, logged = capture_key(database_path)
    if key:
        return key, "captured from pyrekordbox (verified: database opens with it)"
    for candidate_key, key_source in ((logged, "pyrekordbox debug log"), (builtin_key(), "pyrekordbox built-in key")):
        if candidate_key:
            opened, retry_error = try_open(database_path, candidate_key)
            if opened:
                return candidate_key, f"from {key_source} (verified)"
    # nothing verified: if the failure was only a missing sqlcipher, still use the key we have
    candidate_key = logged or builtin_key()
    if candidate_key and "sqlcipher" in (error or "").lower():
        return candidate_key, "from pyrekordbox but NOT verified (sqlcipher3 missing: pip install sqlcipher3-wheels)"
    return None, error or "unknown error"


# ---------------------------------------------------------------------------------------
# QAAC
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def main():
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--force", action="store_true", help="re-detect and overwrite values that already work")
    argument_parser.add_argument("--no-search", action="store_true", help="skip the disk search for qaac")
    argument_parser.add_argument("--env", type=Path, default=ENV_PATH, help="target .env (default: next to this script)")
    arguments = argument_parser.parse_args()

    if arguments.env.exists():
        text, source_file = arguments.env.read_text(encoding="utf-8"), arguments.env
    elif TEMPLATE_PATH.exists():
        text, source_file = TEMPLATE_PATH.read_text(encoding="utf-8"), TEMPLATE_PATH
    else:
        text, source_file = "", None
    crlf = "\r\n" in text
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    current_values = parse_env(text)
    print(f"Starting from: {source_file or 'built-in defaults (no .env or _env found)'}\n")

    found, missing = {}, []

    # --- REKORDBOX_DATABASE_PATH
    database, note = find_database(current_values.get("REKORDBOX_DATABASE_PATH"), arguments.force)
    if database:
        found["REKORDBOX_DATABASE_PATH"] = str(database)
        print(f"[ok]      REKORDBOX_DATABASE_PATH   = {database}\n          ({note})")
        share = database.parent / "share"
        print(f"          share folder {'exists' if share.is_dir() else 'NOT FOUND'}: {share}")
    else:
        missing.append(("REKORDBOX_DATABASE_PATH", f"path of master.db, usually %APPDATA%\\Pioneer\\rekordbox\\master.db ({note})"))
        print(f"[MISSING] REKORDBOX_DATABASE_PATH   ({note})")

    # --- REKORDBOX_ENCRYPTION_KEY
    if database:
        key, knote = find_key(database, current_values.get("REKORDBOX_ENCRYPTION_KEY"), arguments.force)
        if key:
            found["REKORDBOX_ENCRYPTION_KEY"] = key
            flag = "ok" if "NOT verified" not in knote else "WARN"
            print(f"[{flag:<7}] REKORDBOX_ENCRYPTION_KEY  = {key}\n          ({knote})")
        else:
            missing.append(("REKORDBOX_ENCRYPTION_KEY", f"database would not open without/with the key: {knote}"))
            print(f"[MISSING] REKORDBOX_ENCRYPTION_KEY  ({knote})")
    else:
        if current_values.get("REKORDBOX_ENCRYPTION_KEY"):
            found["REKORDBOX_ENCRYPTION_KEY"] = current_values["REKORDBOX_ENCRYPTION_KEY"]
            print(f"[WARN]    REKORDBOX_ENCRYPTION_KEY  kept from .env but not verified (no master.db found)")
        else:
            missing.append(("REKORDBOX_ENCRYPTION_KEY", "needs REKORDBOX_DATABASE_PATH first; key is auto-detected once master.db is found"))
            print("[MISSING] REKORDBOX_ENCRYPTION_KEY  (needs REKORDBOX_DATABASE_PATH first)")

    # --- tools on PATH (not in .env, but the scripts need them)
    for tool in ("ffmpeg", "ffprobe"):
        tool_path = shutil.which(tool)
        if tool_path:
            print(f"[ok]      {tool} on PATH: {tool_path}")
        else:
            missing.append((tool, "not on PATH. Install ffmpeg (includes ffprobe) and add its bin folder to PATH."))
            print(f"[MISSING] {tool} not on PATH")

    # --- write .env
    for setting_name, value in found.items():
        lines = set_value(lines, setting_name, value)
    for setting_name, default, comment in DEFAULTS:
        if setting_name not in current_values and not any(re.match(rf"^\s*{setting_name}\s*=", line_text) for line_text in lines):
            lines += ["", *comment.split("\n"), f"{setting_name}={default}"]
    for setting_name in REQUIRED:                               # keep the key present (empty) for easy hand-editing
        if not any(re.match(rf"^\s*{setting_name}\s*=", line_text) for line_text in lines):
            lines.insert(1 if lines else 0, f"{setting_name}=")
    if arguments.env.exists():
        shutil.copy2(arguments.env, arguments.env.with_name(arguments.env.name + ".bak"))
    arguments.env.write_text(("\r\n" if crlf or os.name == "nt" else "\n").join(lines) + ("\r\n" if crlf or os.name == "nt" else "\n"),
                     encoding="utf-8", newline="")
    print(f"\nWrote {arguments.env}" + (f" (previous version saved as {arguments.env.name}.bak)" if arguments.env.with_name(arguments.env.name + '.bak').exists() else ""))

    # --- what is left
    if missing:
        print("\nSTILL TO FILL IN BY HAND:")
        for setting_name, reason in missing:
            label = f"  {setting_name}=" if setting_name in REQUIRED else f"  {setting_name}"
            print(f"{label:<12} {reason}")
        print(f"\nEdit {arguments.env.name}, then re-run this script to verify, or just run predict_beatgrids.py.")
        sys.exit(1)
    print("\nSUCCESSFULLY FILLED .ENV FILE. You can now run predict_beatgrids_and_cues.py")


if __name__ == "__main__":
    main()