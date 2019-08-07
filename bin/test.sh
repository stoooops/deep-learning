#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

EPOCHS=2

"${SCRIPT_DIR}"/train.sh -m $1 -i 0 -e "${EPOCHS}"

echo
echo
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo
echo

"${SCRIPT_DIR}"/inference.sh -m $1 -e "${EPOCHS}" -w 10 -t 10 -r 1
