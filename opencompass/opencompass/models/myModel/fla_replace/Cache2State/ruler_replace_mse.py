import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from transformers import LlamaConfig, LlamaForCausalLM
from modeling_llama import LlamaConfig, LlamaForCausalLM
import torch
from transformers import AutoTokenizer
from torch.nn import MSELoss
from tqdm import tqdm
import json
from collections import defaultdict

ruler_loss_from_last_layer = {}

file_path = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/opencompass_eval_result/performer-replace-single-layer/20250326_160024/predictions/llama3.2-3b'
for json_file in os.listdir(file_path):
    task_name = json_file.split('.')[0]
    print(f"Processing task {task_name}...")
    data = json.load(open(os.path.join(file_path, json_file)))
    data = data.values()
    data = [d['origin_prompt'] for d in data]
    all_loss_from_last_layer = {}
    for sample in tqdm(data):
        for idx in tqdm(range(28)):
            replaced_layer = [idx]

            MODEL_PATH = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B'

            config = LlamaConfig.from_pretrained(MODEL_PATH)
            config.replaced_layers = replaced_layer
            config.save_output = True
            config.target_layer_type = 'performer'
            config.feature_map = 'performer'

            max_new_tokens = 100

            print("加载替换后模型...")
            model = LlamaForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=torch.bfloat16, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

            model.eval()
            model.generation_config.do_sample = False

            # print(f"{model=}")

            # 生成文本
            # input_text = "你" * 32 * 1024
            input_text = sample
            inputs = tokenizer(input_text, return_tensors='pt')
            print(f"input_length = {inputs['input_ids'].shape[1]}")

            # 移动输入到与模型相同的设备
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成配置
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id避免警告
            )

            layers_outputs = []
            for layer in model.model.layers:
                layers_outputs.append(layer.outputs)
            
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id避免警告
            )
            # 解码输出
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            # print(decoded_output)


            print("加载原始模型")
            config = LlamaConfig.from_pretrained(MODEL_PATH)
            config.save_output = True
            model = LlamaForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=torch.bfloat16, device_map='auto')

            # 生成配置
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
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
            loss_across_layers = [float(l) for l in loss_across_layers]
            res = {}
            res['layer'] = idx
            res['last_layer_loss'] = loss_across_layers[-1]
            res['detailed_layer_loss'] = loss_across_layers
            print(res)
            if idx not in all_loss_from_last_layer:
                all_loss_from_last_layer[idx] = res
            else:
                all_loss_from_last_layer[idx]['last_layer_loss'] += res['last_layer_loss']
                all_loss_from_last_layer[idx]['all_loss_from_last_layer'] += res['all_loss_from_last_layer']
        for idx, res in all_loss_from_last_layer.items():
            all_loss_from_last_layer[idx]['last_layer_loss'] /= len(data)
            all_loss_from_last_layer[idx]['all_loss_from_last_layer'] /= len(data)
    with open(f'./ruler-32k-loss/{task_name}.json', 'w') as f:
        json.dump(all_loss_from_last_layer, f, ensure_ascii=False, indent=4)
    # ruler_loss_from_last_layer[task_name] = all_loss_from_last_layer

        # json.dump(all_loss_from_last_layer, open('./loss_by_layer.json', 'w'), ensure_ascii=False, indent=4)