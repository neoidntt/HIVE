#!/bin/bash
# Double-click this file in Finder to launch the HIVE web app in your browser.
# The first start installs what HIVE needs into the .venv folder next to this file
# (a few minutes); later starts open straight away. Leave the Terminal window open while
# you use the app; close it or press Ctrl+C to stop.

# Run from this script's own folder, wherever it lives.
cd "$(dirname "$0")" || exit 1

# A double-clicked shell may not have pyenv/Homebrew on PATH, so try the usual places.
for py in python3 "$HOME/.pyenv/shims/python3" /opt/homebrew/bin/python3 \
          /usr/local/bin/python3 /usr/bin/python3; do
    if command -v "$py" >/dev/null 2>&1; then
        "$py" launch_hive.py "$@"
        status=$?
        [ $status -ne 0 ] && { echo; read -r -n 1 -p "Press any key to close."; }
        exit $status
    fi
done

echo "ERROR: Python 3 was not found. Install it from https://www.python.org/downloads/ and try again."
read -r -n 1 -p "Press any key to close."
exit 1
