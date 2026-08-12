#!/bin/bash
# Double-click this file in Finder to launch the HIVE web app in your browser.
# It opens a Terminal window (leave it open while you use the app) and stops the
# server when you close that window or press Ctrl+C.

# Run from this script's own folder, wherever it lives.
cd "$(dirname "$0")" || exit 1

echo "======================================================"
echo "  HIVE — High-Processing Imaging Visualization Engine"
echo "======================================================"
echo "  project: $(pwd)"
echo ""

# Find a Python that can actually run the app. pyenv shims may not be on PATH in a
# double-clicked shell, so try the common locations before giving up.
find_python() {
    for py in python3 \
              "$HOME/.pyenv/shims/python3" \
              /opt/homebrew/bin/python3 \
              /usr/local/bin/python3 \
              /usr/bin/python3; do
        if command -v "$py" >/dev/null 2>&1 && "$py" -c "import streamlit, hive" >/dev/null 2>&1; then
            echo "$py"
            return 0
        fi
    done
    return 1
}

PYTHON="$(find_python)"
if [ -z "$PYTHON" ]; then
    echo "ERROR: could not find a Python with HIVE's dependencies installed."
    echo ""
    echo "Set it up once with:"
    echo "    cd \"$(pwd)\""
    echo "    python3 -m pip install -r requirements.txt"
    echo ""
    echo "Press any key to close."
    read -r -n 1
    exit 1
fi

echo "  python : $PYTHON"
echo "  opening http://localhost:8501 in your browser..."
echo "  (leave this window open; close it or press Ctrl+C to stop)"
echo ""

# maxUploadSize comes from .streamlit/config.toml (1 GB). --server.headless=false
# lets Streamlit open the browser tab for you.
exec "$PYTHON" -m streamlit run app.py --server.headless=false
