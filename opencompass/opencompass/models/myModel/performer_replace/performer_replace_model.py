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
# from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
# from .Cache2State.modeling_llama import LlamaConfig, LlamaForCausalLM
from .Cache2State_old.modeling_llama import LlamaConfig, LlamaForCausalLM

@MODELS.register_module()
class PerformerReplaced_LlamaForCausalLM(HuggingFaceCausalLM):
    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        
        replaced_layers = model_kwargs.pop('replaced_layers', [])

        self._set_model_kwargs_torch_dtype(model_kwargs)

        config = LlamaConfig.from_pretrained(path)
        config.replaced_layers = replaced_layers
        config.target_layer_type = 'performer'

        self.model = LlamaForCausalLM.from_pretrained(path, config=config, **model_kwargs)
        # tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        self.model.eval()
        self.model.generation_config.do_sample = False