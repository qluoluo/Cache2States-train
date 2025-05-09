def set_seed(seed, deterministic=True):
    """
    设置所有随机种子以保证实验可重复性
    
    参数：
        seed (int): 随机种子值
        deterministic (bool): 是否启用确定性模式（可能影响性能）
    """
    import random
    import numpy as np
    import torch
    import os
    
    # 设置Python内置随机种子
    random.seed(seed)
    
    # 设置Python哈希种子（防止哈希随机性影响结果）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 设置NumPy随机种子
    np.random.seed(seed)
    
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    
    # 设置CUDA随机种子（适用于所有GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 设置cuDNN参数以保证确定性
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        
        # 添加额外的CUDA确定性设置（PyTorch 1.7+）
        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(deterministic)
    
    # 针对多进程/分布式训练的可选设置
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # 日志输出
    print(f"Set all random seeds to {seed} [Deterministic: {deterministic}]")

set_seed(59, deterministic=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from transformers import LlamaConfig, LlamaForCausalLM
from modeling_llama import LlamaConfig, LlamaForCausalLM
import torch
from transformers import AutoTokenizer

MODEL_PATH = '/remote-home1/share/models/llama3_2_hf/Llama-3.2-3B/'

config = LlamaConfig.from_pretrained(MODEL_PATH)
config.replaced_layers = [27]
config.target_layer_type = 'performer'
config.feature_map = 'performer'
# config.feature_map = 't2r'
# config.do_feature_map_norm = True

print("加载模型...")
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

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
    max_new_tokens=100,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id避免警告
)

# 解码输出
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)