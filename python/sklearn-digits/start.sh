#!/usr/bin/env bash

ECHO_PREFIX="[rekcurd example: python sklearn-digits]: "

set -e
set -u

echo "$ECHO_PREFIX Start.."

pip install -r requirements.txt
python sample_model_build.py
python server.py