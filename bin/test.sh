#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

${SCRIPT_DIR}/train.sh -m $1 -e 1 -i 0
${SCRIPT_DIR}/inference.sh -m $1 -e 1 -w 10 -t 10 -r 1 -i
