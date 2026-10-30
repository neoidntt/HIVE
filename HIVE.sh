#!/bin/bash
# Start HIVE on Linux (or any shell). The first start installs what HIVE needs.
# Make it double-clickable once with:  chmod +x HIVE.sh
cd "$(dirname "$0")" || exit 1
for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        exec "$py" launch_hive.py "$@"
    fi
done
echo "Python 3.10 or newer was not found. Install it with your package manager, then run this again."
read -r -p "Press Enter to close." _
exit 1
