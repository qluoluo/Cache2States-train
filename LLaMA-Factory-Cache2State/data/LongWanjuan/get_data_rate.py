import os
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 配置参数
input_root = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/data/OpenLMLab___LongWanjuan"
output_dir = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/data/sampled_LongWanjuan"
total_num = 10000
ratio_dict = {
    'ChinaNews-cn': 0.036945358423435035, 'Law-cn': 0.21959819815778192, 
    'Patent-cn': 0.25097437286506313, 'TextBook-cn': 0.2981288662267506, 
    'WebText-cn': 0.19435320432696937, 'RedPajamaCommonCrawl': 0.67 * 9, 
    'RedPajamaC4': 0.15 * 9, 'RedPajamaArXiv': 0.025 * 9, 
    'RedPajamaBook': 0.045 * 9, 'RedPajamaWikipedia': 0.045 * 9, 
    'RedPajamaStackExchange': 0.02 * 9, 'RedPajamaGithub': 0.045 * 9
}
root_ratio_dict = {
    'aggregated': 0.5,
    'holistic': 0.5,
    'chaotic': 0,
}

# 归一化比例
sum_value = sum(ratio_dict.values())
ratio_dict = {k: v/sum_value for k, v in ratio_dict.items()}
print(f"{ratio_dict=}")

print("🎬 开始数据处理流程")

# 查找所有符合条件的数据路径
valid_paths = []
for root, dirs, _ in os.walk(input_root):
    rel_path = os.path.relpath(root, input_root)
    path_parts = rel_path.split(os.sep)
    if len(path_parts) >= 2:
        root_category = path_parts[-2]
        ratio_key = path_parts[-1]
        if (root_category in root_ratio_dict and 
            root_ratio_dict[root_category] > 0 and 
            ratio_key in ratio_dict):
            valid_paths.append((root, ratio_key, root_category))

if not valid_paths:
    print("❌ 未找到任何有效数据路径")
    exit()

# 计算路径权重
total_weight = sum(ratio_dict[rk] * root_ratio_dict[rc] for _, rk, rc in valid_paths)
path_weights = {rp: ratio_dict[rk] * root_ratio_dict[rc]
               for rp, rk, rc in valid_paths}

print(path_weights)

output_dir = os.path.dirname(__file__)
output_fp = os.path.join(output_dir, "data_rate.json")

with open(output_fp, 'w') as f:
    json.dump(path_weights, f, indent=4)