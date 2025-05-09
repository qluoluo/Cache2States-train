# import os
import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import SlurmRunner, LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

from opencompass.models.myModel.hf_strip_model import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM
# from opencompass.models.myModel.performer_replace.performer_replace_model import PerformerReplaced_LlamaForCausalLM
from opencompass.models.myModel.fla_replace.fla_replace_model import FlaReplaced_LlamaForCausalLM

from mmengine.config import read_base

with read_base():
    # from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    # from opencompass.configs.datasets.ruler.ruler_32k_gen_niah_single import ruler_datasets as ruler_datasets_niah_single
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k

datasets = []
# datasets += ruler_datasets_niah_single
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k
datasets += ruler_datasets_32k


model_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/models/Llama-3_2-3B-fla-replace4-fuseFull'

default_model_kwargs = dict(device_map='cuda', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                # rope_scaling={"type": "dynamic", "factor": 4.0}),
                                attn_implementation='flash_attention_2',
                            )
                            
max_seq_len=32*1024
max_out_len=50
run_cfg=dict(num_gpus=1, num_procs=1)
batch_size=1

models = []

models.append(dict(
        type=HuggingFaceCausalLM,
        abbr='base',
        path=model_path,
    ))

work_dir = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/opencompass_eval_result/hedgehog-fuse'

ckpt_root = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/Llama-3_2-3B-fla-replace4-fuseFull-hedgehog-freeze-pt-lr1e-3'

ckpt_list = [
        'checkpoint-200',
        'checkpoint-400',
        'checkpoint-600',
        'checkpoint-800',
        'checkpoint-992',
    ]

for ckpt_fp in ckpt_list:
    models.append(dict(
            type=HuggingFaceCausalLM,
            abbr=ckpt_fp.replace('checkpoint', 'ckpt'),
            # abbr=ckpt_fp,
            # abbr='llama3.2-3b' + '-' + ckpt_fp,
            # model_kwargs=dict(replaced_layers=[]),
            path=ckpt_root + '/' + ckpt_fp,
        ))



for model in models:
    model.update(dict(
        # path=model_path,
        model_kwargs=default_model_kwargs | model.get('model_kwargs', {}),
        tokenizer_path=model_path,
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        run_cfg=run_cfg,
        batch_size=batch_size,
    ))
    # print(model)

# print(f"{models=}")

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=7,
        task=dict(type=OpenICLInferTask),
        retry=1),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask)),
)