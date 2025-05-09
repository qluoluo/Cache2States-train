import random
import numpy as np
import os
import torch
from tqdm import tqdm
from torch.nn import MSELoss
import json

os.environ["CUDA_VISIBLE_DEVICES"]="7"

def fix_random_seeds(seed=42):
    """
    固定主流随机数生成库的种子以实现可重复性
    参数：
        seed: 整型，默认42，所有随机种子将基于此值设置
    """
    # 设置Python内置随机模块
    random.seed(seed)
    
    # 设置NumPy随机种子
    np.random.seed(seed)
    
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        # 保证卷积结果确定性（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置Python哈希种子（影响哈希随机化）
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_random_seeds(42)
print("所有随机种子已固定为42")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

model_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/models/Llama-3.2-3B-fla-hybrid-copy'
# model_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B'
print(f"{os.path.basename(model_path)=}")

all_loss_from_last_layer = []
nb_features = 512

for idx in tqdm(range(28)):
    replaced_layer = [idx]

    config = LlamaConfig.from_pretrained(model_path)
    config.replaced_layers = replaced_layer
    config.save_output = True

    # print(config)
    # config.mode = "chunk"

    config.q_nb_features = nb_features
    config.k_nb_features = nb_features

    max_new_tokens = 100

    print("加载替换后模型...")
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    model.generation_config.do_sample = False

    # print(f"{model=}")

    # 生成文本
    # input_text = "你" * 32 * 1024
    input_text = "User: Please introduce yourself.\nAssistant:"
    inputs = tokenizer(input_text, return_tensors='pt')
    print(f"input_length = {inputs['input_ids'].shape[1]}")

    # 移动输入到与模型相同的设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成配置
    output = model.generate(
        **inputs,
        max_new_tokens=1,
        repetition_penalty=1.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id避免警告
    )

    layers_outputs = []
    for layer in model.model.layers:
        layers_outputs.append(layer.outputs)
    
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id避免警告
    )
    # 解码输出
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)


    print("加载原始模型")
    config = LlamaConfig.from_pretrained(model_path)
    config.save_output = True
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)

    # 生成配置
    output = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id避免警告
    )

    # # 解码输出
    # decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(decoded_output)

    origin_layers_outputs = []
    for layer in model.model.layers:
        origin_layers_outputs.append(layer.outputs)
    loss_across_layers = []
    loss_fn = MSELoss()
    for o1, o2 in zip(layers_outputs, origin_layers_outputs):
        loss_across_layers.append(loss_fn(o1[0], o2[0]))
    print(loss_across_layers)
    all_loss_from_last_layer.append(
        {
            'layer': idx,
            'loss': float(loss_across_layers[-1]),
            'output': decoded_output
        }
    )

# 生成配置
output = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id避免警告
)

# 解码输出
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)

all_loss_from_last_layer.insert(0,
    {
        'layer': -1,
        'loss': 0.0,
        'output': decoded_output
    }
)
json.dump(all_loss_from_last_layer, open(f'./loss_by_layer-feature-{nb_features}-penalty-{1.1}.json', 'w'), ensure_ascii=False, indent=4)