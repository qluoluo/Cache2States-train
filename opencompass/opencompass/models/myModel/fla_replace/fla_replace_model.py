import random
import numpy as np
import os
import torch

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

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

# from .huggingface import HuggingFaceCausalLM
# from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.models.myModel.hf_strip_model import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM

# from .Cache2State.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

@MODELS.register_module()
class FlaReplaced_LlamaForCausalLM(HuggingFaceCausalLM):
    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):

        fix_random_seeds(42)
        print("所有随机种子已固定为42")
        
        replaced_layers = model_kwargs.pop('replaced_layers', [])
        q_nb_features = model_kwargs.pop('q_nb_features', 256)
        k_nb_features = model_kwargs.pop('k_nb_features', 256)

        self._set_model_kwargs_torch_dtype(model_kwargs)

        config = LlamaConfig.from_pretrained(path)
        config.replaced_layers = replaced_layers
        config.target_layer_type = 'performer'
        config.feature_map = 'performer'
        config.q_nb_features = q_nb_features
        config.k_nb_features = k_nb_features

        # self.model = LlamaForCausalLM.from_pretrained(path, config=config, **model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(path, config=config, **model_kwargs)

        # print(self.model)

        self.model.eval()
        self.model.generation_config.do_sample = False

        # tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # input_text = "User: Please introduce yourself.\nAssistant:"
        # inputs = tokenizer(input_text, return_tensors='pt')
        # inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # # 生成配置
        # output = self.model.generate(
        #     **inputs,
        #     max_new_tokens=100,
        #     do_sample=False,
        #     pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id避免警告
        # )

        # # 解码输出
        # decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        # print(decoded_output)

        # exit()

    # def generate(self,
    #              inputs: List[str],
    #              **kwargs) -> List[str]:
    #     if hasattr(self.model, 'clear_cache'):
    #         self.model.clear_cache()
    #     inputs = [x.strip() for x in inputs]
    #     return super().generate(inputs, **kwargs)