#!/usr/bin/env bash

ECHO_PREFIX="[rekcurd example]: "

set -e
set -u

echo "$ECHO_PREFIX Start.."

cd python/sklearn-digits
sh start.sh
