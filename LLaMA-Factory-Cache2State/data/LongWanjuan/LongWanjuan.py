import json
import os
import datasets

import random
from pathlib import Path
import warnings

# _HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
_DESCRIPTION = "Human preference data about helpfulness and harmlessness."
# _CITATION = ""
# _HOMEPAGE = f"{_HF_ENDPOINT}/datasets/Anthropic/hh-rlhf"
# _LICENSE = "mit"
# _URL = f"{_HF_ENDPOINT}/datasets/Anthropic/hh-rlhf/resolve/main/"
# _URLS = {
#     "train": [
#         _URL + "harmless-base/train.jsonl.gz",
#         _URL + "helpful-base/train.jsonl.gz",
#         _URL + "helpful-online/train.jsonl.gz",
#         _URL + "helpful-rejection-sampled/train.jsonl.gz",
#     ],
#     "test": [
#         _URL + "harmless-base/test.jsonl.gz",
#         _URL + "helpful-base/test.jsonl.gz",
#         _URL + "helpful-online/test.jsonl.gz",
#         _URL + "helpful-rejection-sampled/test.jsonl.gz",
#     ],
# }


class DataSampler:
    def get_data_rate(self, rate_fp):
        with open(rate_fp, "r") as f:
            data_rate = json.load(f)
        return data_rate

    def __init__(self):
        rate_fp = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/LLaMA-Factory-Cache2State/data/LongWanjuan/data_rate.json'
        self.folder_probs = self.get_data_rate(rate_fp)
        self.generators = {}
        for folder in self.folder_probs:
            self.generators[folder] = self._infinite_jsonl_generator(folder)
    
    def _infinite_jsonl_generator(self, folder):
        # 按文件名顺序获取所有jsonl文件
        files = sorted(Path(folder).glob("*.jsonl"))
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    yield json.loads(line)['content']
        
        while True:
            warnings.warn(f"######### file used up in {folder} #########", RuntimeWarning)
            yield None
    
    def sample(self):

        while len(list(self.generators.keys())) > 0:
            
            ret_data = None
            while ret_data is None:
                folders = list(self.folder_probs.keys())
                probabilities = list(self.folder_probs.values())

                chosen_folder = random.choices(folders, weights=probabilities, k=1)[0]
                ret_data = next(self.generators[chosen_folder])

                if ret_data is None:
                    self.generators.pop(chosen_folder)
                    self.folder_probs.pop(chosen_folder)
            
            yield ret_data


class LongWanjuan(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, # homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )

    # def _split_generators(self, dl_manager: datasets.DownloadManager):
    #     file_path = dl_manager.download_and_extract(_URLS)
    #     return [
    #         datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": file_path["train"]}),
    #         datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": file_path["test"]}),
    #     ]
    def _split_generators(self, dl_manager: datasets.DownloadManager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={})]

    # def _generate_examples(self, filepaths: list[str]):
    def _generate_examples(self):
        key = 0
        sampler = DataSampler()
        for text in sampler.sample():
            if key > 95000:
            # if key > 500:
                return
            key += 1
            yield key, {"text": text}

    def _generate_examples_unlimited(self):
        key = 0
        sampler = DataSampler()
        for text in sampler.sample():
            # if key > 900:
            # if key > 95000:
            # if key > 500:
            #     return
            key += 1
            yield key, {"text": text}

if __name__ == '__main__':
    wanjuan_dataset = LongWanjuan()
    g = wanjuan_dataset._generate_examples('')
    print(next(g))