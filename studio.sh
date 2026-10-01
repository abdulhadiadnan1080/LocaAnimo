#!/bin/sh
# Start the LocaAnimo studio and open it in the browser.
cd "$(dirname "$0")"
PYTHONPATH=src exec .venv/bin/python -m locaanimo.server "$@"
