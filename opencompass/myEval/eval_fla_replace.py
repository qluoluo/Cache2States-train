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
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets

datasets = []
# datasets += ruler_datasets_niah_single
datasets += ruler_datasets

work_dir = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/opencompass_eval_result/fla-replace-single-layer'

model_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B'
default_model_kwargs = dict(device_map='cuda', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                # rope_scaling={"type": "dynamic", "factor": 4.0}),
                                attn_implementation='flash_attention_2',
                            )
                            
max_seq_len=32*1024
max_out_len=50
run_cfg=dict(num_gpus=1, num_procs=1)
batch_size=1

models = []

for i in range(28):
# for i in [27]:
    models.append(dict(
        type=FlaReplaced_LlamaForCausalLM,
        abbr=f'llama3.2-3b-performerReplace-[{i}]',
        model_kwargs=dict(replaced_layers=[i]),
    ),)

# models.append(dict(
#         type=HuggingFaceCausalLM,
#         abbr='llama3.2-3b',
#     ))

# models = [
#     dict(
#         type=PerformerReplaced_LlamaForCausalLM,
#         abbr='llama3.2-3b-performerReplace-[27]',
#         model_kwargs=dict(replaced_layers=[27]),
#     ),
#     dict(
#         type=PerformerReplaced_LlamaForCausalLM,
#         abbr='llama3.2-3b-performerReplace-[26,27]',
#         model_kwargs=dict(replaced_layers=[26,27]),
#     ),
#     dict(
#         type=PerformerReplaced_LlamaForCausalLM,
#         abbr='llama3.2-3b-performerReplace-[25,26,27]',
#         model_kwargs=dict(replaced_layers=[25,26,27]),
#     ),
# ]

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
        # max_num_workers=4,
        task=dict(type=OpenICLInferTask),
        retry=1),
)

# eval = dict(
#     partitioner=dict(type=NaivePartitioner),
#     runner=dict(
#         type=LocalRunner,
#         max_num_workers=32,
#         task=dict(type=OpenICLEvalTask)),
# )