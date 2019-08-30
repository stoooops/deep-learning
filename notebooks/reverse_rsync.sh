#!/usr/bin/env bash

MAKA_DIR=$(ssh "${1}" "env | grep MAKA_CODE | cut -d '=' -f2")
SYNC_TO_DIR="${MAKA_DIR}/deep-learning"

run_rsync() {
    [ -z "${1}" ] && echo "No argument supplied" && return 1
    [ -z "${2}" ] && echo "Must pass two arguments. Only got one." && return 2
    SRC=$(cd "${1}" || exit; pwd)
    rsync -r -a -v \
        --exclude=.DS_Store \
        --exclude=.python-version \
        --include=tmp/.gitignore \
        --include=tmp/logs/.gitignore \
        --include=tmp/logs/tensorboard/.gitignore \
        --exclude=tmp/* \
        --exclude=.idea \
        --exclude=__pycache__ \
        --exclude=.ipynb_checkpoints \
        -e ssh "${SRC}" "${2}"
    echo "=================================================="
}

loop_rsync() {
    [ -z "${1}" ] && echo "No argument supplied" && return 1
    [ -z "${2}" ] && echo "Must pass two arguments. Only got one." && return 2
    SRC=$(cd "${1}" || exit; pwd)
    DEST="${2}:${SYNC_TO_DIR}"
    echo "Running rsync for ${SRC} to ${DEST}"
    run_rsync "${SRC}" "${DEST}"
    echo "Watching for filesystem changes at ${SRC}"
    inotifywait -m . -e close_write |
      while read path action file; do
        echo "Running rsync for ${SRC} to ${DEST}" && run_rsync "${SRC}" "${DEST}";
    done
}


loop_rsync . "${1}"