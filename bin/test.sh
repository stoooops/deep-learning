#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

EPOCHS=2

IFS=',' read -ra ADDR <<< "$1"
for i in "${ADDR[@]}"; do
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


