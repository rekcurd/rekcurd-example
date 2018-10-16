#!/usr/bin/env bash

ECHO_PREFIX="[drucker example]: "

set -e
set -u

echo "$ECHO_PREFIX Start.."

pip install -r requirements.txt
python sample_model_build.py
python server.py
