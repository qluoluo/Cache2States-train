#!/bin/bash
root_path=/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089
source ${root_path}/local_conda.sh

echo "start eval"
wait
conda activate fla
wait
which python
cd ${root_path}/train/opencompass
sleep 5
opencompass myEval/liger/liger-16k-long.py
wait
sleep 5

bash /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/public/cache2state-4-13-eva.sh