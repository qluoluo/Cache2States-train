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
    # from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k

datasets = []
# datasets += ruler_datasets_niah_single
# datasets += ruler_datasets
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k

# model_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B'
model_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/models/Llama-3.2-3B-fla-hybrid'

default_model_kwargs = dict(device_map='cuda', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                # rope_scaling={"type": "dynamic", "factor": 4.0}),
                                attn_implementation='flash_attention_2',
                            )
                            
max_seq_len=32*1024
max_out_len=50
run_cfg=dict(num_gpus=1, num_procs=1)
batch_size=1

models = []

work_dir_root = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/opencompass_eval_result/fla-v0-diff-ruler'

# 替换单层
# for i in range(28):
# for i in [27]:
# for i in [21]:
# for i in [0, 1, 5, 9, 13, 17, 21, 25, 26, 27]:
# for i in [21, 25, 26, 27]:
#     models.append(dict(
#         type=FlaReplaced_LlamaForCausalLM,
#         # abbr=f'llama3.2-3b-performerReplace-[{i}]',
#         abbr=f'[{i}]',
#         model_kwargs=dict(replaced_layers=[i]),
#     ),)
# work_dir = work_dir_root + "/single_layer"


models.append(dict(
    type=FlaReplaced_LlamaForCausalLM,
    # abbr=f'llama3.2-3b-performerReplace-[{i}]',
    abbr=f'[13-27]',
    model_kwargs=dict(replaced_layers=list(range(13,27))),
),)
work_dir = work_dir_root + "/exp"


# 从最顶层开始连续替换i层
# for i in [2,4,7,14]:
# for i in range(2,15):
#     models.append(dict(
#         type=FlaReplaced_LlamaForCausalLM,
#         # abbr=f'llama3.2-3b-performerReplace-[{27-i+1}-{27}]',
#         abbr=f'[{27-i+1}-{27}]',
#         model_kwargs=dict(replaced_layers=list(range(27-i+1, 28))),
#     ),)
# work_dir = work_dir_root + "/to_last_layer"


# 从最顶层开始每隔i层替换
# for i in [2,4,7,14]:
# for i in range(2,20):
#     models.append(dict(
#         type=FlaReplaced_LlamaForCausalLM,
#         # abbr=f'llama3.2-3b-performerReplace-[range(27, 0, {-i})]',
#         abbr=f'27—0-{i}',
#         model_kwargs=dict(replaced_layers=list(range(27, 0, -i))),
#     ),)
# work_dir = work_dir_root + "/last_layer_step"


# 保留底层4层的基础上，从最顶层开始每隔i层替换
# for i in [2,4,6,12]:
# # for i in range(2,14):
#     models.append(dict(
#         type=FlaReplaced_LlamaForCausalLM,
#         # abbr=f'llama3.2-3b-performerReplace-[range(27, 0, {-i})]',
#         abbr=f'27—3-{i}',
#         model_kwargs=dict(replaced_layers=list(range(27, 3, -i))),
#     ),)
# work_dir = work_dir_root + "/last_layer_step_remian_bottom"


# models.append(dict(
#         type=HuggingFaceCausalLM,
#         abbr='llama3.2-3b',
#         model_kwargs=dict(replaced_layers=[]),
#     ))
# work_dir = work_dir_root + "/total"


for model in models:
    model.update(dict(
        path=model_path,
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
        max_num_workers=128,
        task=dict(type=OpenICLEvalTask)),
)