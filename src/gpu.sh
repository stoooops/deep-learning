#!/usr/bin/env bash

NVIDIA_HEADER="timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used"
#echo $NVIDIA_HEADER

nvidia-smi --query-gpu=$NVIDIA_HEADER --format=csv | column -s , -t
while true; do
    sleep 5
    nvidia-smi --query-gpu=$NVIDIA_HEADER --format=csv | column -s , -t | tail -1
done


