#!/usr/bin/env bash

ECHO_PREFIX="[Rekcurd example: python sklearn-digits]: "

set -e
set -u

echo "$ECHO_PREFIX start.."

pip install -r requirements.txt
python sample_model_build.py
python app.py
