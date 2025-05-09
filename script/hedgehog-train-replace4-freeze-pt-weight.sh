#!/bin/bash
root_path=/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089
source ${root_path}/local_conda.sh

# echo "### start train ###"
# which python
# wait
# conda activate llamafactory
# wait
# cd ${root_path}/train/LLaMA-Factory-Cache2State
# wait
# sleep 3
# llamafactory-cli train ./examples/custom/llama3.2_replace4-fuseFull_hedgehog-freeze-pt-weight.yaml
# wait
# echo "### end train ###"
# sleep 60


echo "start eval"
wait
conda activate fla
wait
which python
cd ${root_path}/train/opencompass
sleep 5
opencompass myEval/hedgehog-fuse/eval-replace4-fuseFull_ckpt-freeze-pt-weight.py
wait
sleep 5



echo "start occupy eval"
wait
conda activate fla
wait
which python
cd ${root_path}/train/opencompass
sleep 5
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 opencompass myEval/eval-occupy.py
wait
sleep 5