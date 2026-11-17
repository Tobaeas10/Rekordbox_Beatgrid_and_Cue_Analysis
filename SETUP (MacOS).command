#!/bin/bash
# One-click setup for macOS. Double-click this file.
# Installs Python 3.14 (official installer from python.org) if no Python 3.14 or newer is found, then runs lib/setup_install.py.
cd "$(dirname "$0")" || exit 1
PYVERSION="3.14.10"
PYTHON=""

is_new_enough() {
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 14) else 1)' >/dev/null 2>&1
}

find_python() {
  local candidate found
  for candidate in python3.16 python3.15 python3.14 \
      /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 \
      /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    found="$(command -v "$candidate" 2>/dev/null)" || continue
    [ "$found" = "/usr/bin/python3" ] && continue        # Apple's stub: old, and may open an install dialog
    if is_new_enough "$found"; then PYTHON="$found"; return 0; fi
  done
  return 1
}

install_python() {
  local package
  package="$(mktemp -d)/python-$PYVERSION-macos11.pkg"
  echo "Python 3.14 or newer was not found on this Mac."
  echo "It will now be downloaded from python.org (about 70 MB) and installed with the official installer."
  echo
  curl -fL --progress-bar -o "$package" "https://www.python.org/ftp/python/$PYVERSION/python-$PYVERSION-macos11.pkg" || return 1
  echo
  echo "The installer needs your Mac password (the one you log in with)."
  echo "Nothing is shown while you type it. Press Enter when done."
  sudo installer -pkg "$package" -target / || return 1
  rm -f "$package"
  PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14"
  [ -x "$PYTHON" ] && is_new_enough "$PYTHON"
}

wait_for_key() {
  echo
  read -n 1 -s -r -p "Press any key to close this window ... "
  echo
}

if ! find_python; then
  if ! install_python; then
    echo
    echo "Python could not be installed. Check your internet connection and open this file again,"
    echo "or install Python 3.14 yourself from https://www.python.org/downloads/ and open this file again."
    wait_for_key
    exit 1
  fi
  echo "Python $PYVERSION is installed."
  echo
fi

echo "Using Python: $PYTHON"
echo
"$PYTHON" -m lib.setup_install --no-pause "$@"
status=$?
wait_for_key
exit $status
