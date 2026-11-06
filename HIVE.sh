#!/bin/bash
# Start HIVE on Linux (and macOS). Double-click it, or run  ./HIVE.sh
#
# First start: downloads what HIVE needs into this folder (the installer "uv", a private Python
# if the computer has none, and the libraries in requirements.txt) and then opens HIVE in your
# browser. Later starts skip the download and open straight away. Nothing is installed
# system-wide; deleting this folder removes everything.
#
# The first time, it asks whether to add a HIVE shortcut to your desktop (Yes / No). Answer Yes and a HIVE icon
# appears on the desktop; double-click it to start HIVE from then on. Answer No and it asks whether to remind you
# next time. The answer is kept in .hive_settings.json.
#
# If your file manager shows this file in a text editor instead of running it, open it once with
# right-click > "Run as a Program" (or ./HIVE.sh in a terminal).
here="$(cd "$(dirname "$0")" && pwd)" || exit 1
cd "$here" || exit 1
self="$here/${0##*/}"
export HIVE_LAUNCHER="$self"        # the desktop shortcut should start this file
tools="$here/.hive_tools"
UV_DOWNLOAD="https://github.com/astral-sh/uv/releases/latest/download/"
PYTHON_VERSION="3.12"

pause() { read -r -p "Press Enter to close this window." _ </dev/tty 2>/dev/null || true; }

# --- 1. Started without a terminal (double-click from a file manager)? Open one, so the
# download progress and any error message are visible, and run this script again inside it.
has_display() { [ -n "$DISPLAY$WAYLAND_DISPLAY" ] || [ "$(uname -s)" = Darwin ]; }
open_in_terminal() {
    if [ "$(uname -s)" = Darwin ]; then
        open -a Terminal "$self" && return 0
        return 1
    fi
    local term
    for term in x-terminal-emulator gnome-terminal kgx konsole xfce4-terminal mate-terminal \
                lxterminal tilix kitty alacritty xterm; do
        command -v "$term" >/dev/null 2>&1 || continue
        case "$term" in
            gnome-terminal|kgx|tilix) "$term" -- bash "$self" "$@" ;;
            xfce4-terminal|mate-terminal) "$term" -x bash "$self" "$@" ;;
            *) "$term" -e bash "$self" "$@" ;;
        esac
        return 0
    done
    return 1
}
if [ ! -t 1 ] && [ -z "$HIVE_IN_TERMINAL" ] && has_display; then
    export HIVE_IN_TERMINAL=1
    open_in_terminal "$@" && exit 0
fi

# --- 2. A Python that can run the launcher (it only needs the standard library).
PYTHON=""
for py in python3 python python3.13 python3.12 python3.11 python3.10 "$HOME/.pyenv/shims/python3" \
          /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    if command -v "$py" >/dev/null 2>&1 &&
       "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        PYTHON="$py"; break
    fi
done

fetch() {   # fetch URL FILE
    if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 2 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
    else echo "Neither curl nor wget is installed, so the installer cannot be downloaded." >&2; return 1
    fi
}

# --- 3. No usable Python: fetch the installer (uv) ourselves. uv then brings its own Python.
install_uv() {
    local asset tmp expected actual
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)            asset="uv-x86_64-unknown-linux-gnu.tar.gz" ;;
        Linux-aarch64|Linux-arm64) asset="uv-aarch64-unknown-linux-gnu.tar.gz" ;;
        Darwin-arm64)            asset="uv-aarch64-apple-darwin.tar.gz" ;;
        Darwin-x86_64)           asset="uv-x86_64-apple-darwin.tar.gz" ;;
        *) echo "No automatic installer for $(uname -s)/$(uname -m). Install Python 3.10 or newer, then run this again." >&2
           return 1 ;;
    esac
    mkdir -p "$tools" && tmp="$(mktemp -d "$tools/download.XXXXXX")" || return 1
    echo "Python was not found, so the installer (uv) is downloaded first ..."
    fetch "$UV_DOWNLOAD$asset" "$tmp/$asset" && fetch "$UV_DOWNLOAD$asset.sha256" "$tmp/$asset.sha256" || {
        echo "Could not download the installer. Check the internet connection, or install Python 3.10+." >&2
        rm -rf "$tmp"; return 1; }
    expected="$(awk '{print $1; exit}' "$tmp/$asset.sha256")"
    if command -v sha256sum >/dev/null 2>&1; then actual="$(sha256sum "$tmp/$asset" | awk '{print $1}')"
    else actual="$(shasum -a 256 "$tmp/$asset" | awk '{print $1}')"; fi
    if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
        echo "The downloaded installer failed its checksum test, so it was not used." >&2
        rm -rf "$tmp"; return 1
    fi
    tar -xzf "$tmp/$asset" -C "$tmp" && mv "$tmp"/*/uv "$tools/uv" && chmod +x "$tools/uv"
    local status=$?
    rm -rf "$tmp"
    return $status
}

if [ -n "$PYTHON" ]; then
    "$PYTHON" launch_hive.py "$@"
    status=$?
else
    UV="$(command -v uv)"
    [ -n "$UV" ] || { [ -x "$tools/uv" ] || install_uv; UV="$tools/uv"; }
    if [ ! -x "$UV" ]; then
        echo "HIVE could not be started: no Python 3.10+ and the installer could not be fetched." >&2
        pause; exit 1
    fi
    # Everything uv fetches stays inside this folder, exactly as the launcher does for the libraries.
    UV_CACHE_DIR="$tools/cache" UV_PYTHON_INSTALL_DIR="$tools/python" NO_COLOR=1 \
        "$UV" run --no-project --python "$PYTHON_VERSION" launch_hive.py "$@"
    status=$?
fi

# The launcher waits for Enter itself after a problem it explains (exit 1) and on Ctrl+C (130).
case "$status" in 0|1|130|143) ;; *) echo "HIVE stopped unexpectedly (exit code $status)."; pause ;; esac
exit "$status"
