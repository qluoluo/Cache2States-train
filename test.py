import random

import numpy as np
import tiktoken
from datasets import Dataset
from transformers import AutoTokenizer

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

# tokenizer = tiktoken.encoding_for_model('gpt-4')

# print([6:28:7])
print(list(range(27, 0, -14)))
print(list(range(27, 0, -7)))