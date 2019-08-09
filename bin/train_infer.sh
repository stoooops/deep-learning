#!/bin/bash

set -e
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MODELS="${1:-basic}"
EPOCHS="${2:-2}"

IFS=',' read -ra ADDR <<< "$MODELS"
for i in "${ADDR[@]}"; do
    cmd=
    "${SCRIPT_DIR}"/train.sh -m $i -i 0 -e "${EPOCHS}"

    echo
    echo
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo
    echo

    "${SCRIPT_DIR}"/inference.sh -m $i -e "${EPOCHS}" -w 10 -t 10 -r 1

    echo
    echo
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo
    echo
done


