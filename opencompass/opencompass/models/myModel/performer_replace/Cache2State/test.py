import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_llama import LlamaForCausalLM, LlamaConfig
from fla.models.linear_attn import LinearAttentionForCausalLM
from performer_pytorch import Performer, FastAttention
from modeling_performer import fast_attention_forward, repeat_kv
import json
from tqdm import tqdm
import os
import numpy as np
import glob
import argparse

def print_gpu_usage():
    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated(device)/1024**2
        peak = torch.cuda.max_memory_allocated(device)/1024**2
        return f"{current:.1f}MB / {peak:.1f}MB"
    return "N/A"

def process_task(task_path, model, tokenizer, device, max_samples=None):
    """处理单个任务文件，返回该任务的层损失"""
    print(f"处理任务: {os.path.basename(task_path)}")
    
    # 加载数据
    with open(task_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        # 随机选择指定数量的样本
        import random
        sample_ids = list(data.keys())
        random.shuffle(sample_ids)
        sample_ids = sample_ids[:max_samples]
        data = {k: data[k] for k in sample_ids}
    
    print(f"加载了 {len(data)} 个样本")
    
    # 存储结果的数据结构
    all_losses = {}          # 存储每个样本每层的损失
    layer_avg_losses = {}    # 存储每层在所有样本上的平均损失
    sample_avg_losses = {}   # 存储每个样本在所有层上的平均损失
    processed_samples = 0    # 已处理的样本数
    
    # 处理每个样本
    for sample_id, item in tqdm(list(data.items()), desc="处理样本"):
        prompt = item['origin_prompt']
        try:
            # 编码输入文本
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            input_length = inputs.input_ids.shape[1]
            
            print(f"样本 {sample_id} | 输入长度: {input_length} | GPU: {print_gpu_usage()}")
            
            # forward 调用，获取所有层的输出
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    return_dict=True
                )
            
            # 初始化当前样本的损失映射
            sample_losses = {}
            
            # 处理每一层
            layer_count = len(model.model.layers)
            for idx, layer in enumerate(model.model.layers):
                # 获取QKV状态
                loss_value = layer.self_attn.performer_mse_loss
                
                # 记录此层的损失
                sample_losses[idx] = loss_value
                
                # 累积到层平均损失
                if idx not in layer_avg_losses:
                    layer_avg_losses[idx] = 0.0
                layer_avg_losses[idx] += loss_value
            
            # 存储这个样本的所有层损失
            all_losses[sample_id] = sample_losses
            
            # 计算这个样本在所有层上的平均损失
            sample_avg_losses[sample_id] = sum(sample_losses.values()) / len(sample_losses)
            
            processed_samples += 1
            
            # 清理GPU内存
            del outputs, inputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        except Exception as e:
            print(f"处理样本 {sample_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 计算每层的平均损失
    if processed_samples > 0:
        for layer_idx in layer_avg_losses:
            layer_avg_losses[layer_idx] /= processed_samples
    
    # 计算整体平均损失
    overall_avg_loss = np.mean([loss for loss in sample_avg_losses.values()]) if sample_avg_losses else 0
    
    return {
        "task_name": os.path.basename(task_path),
        "all_sample_layer_losses": all_losses,
        "layer_average_losses": layer_avg_losses,
        "sample_average_losses": sample_avg_losses,
        "overall_average_loss": overall_avg_loss,
        "processed_samples": processed_samples,
        "total_samples": len(data)
    }

def main():
    parser = argparse.ArgumentParser(description="计算LLama模型在RULER任务上的Performer注意力损失")
    parser.add_argument("--model_path", type=str, default="/remote-home1/share/models/llama3_2_hf/Llama-3.2-3B/",
                        help="模型路径")
    parser.add_argument("--data_dir", type=str, default="/remote-home1/mqhuang/mttt/RULER/scripts/data/niah/llama3.1-8b/",
                        help="数据目录路径")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="输出目录")
    parser.add_argument("--context_length", type=str, default="32k",
                        help="要处理的上下文长度，例如: 4k,8k,16k,32k")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="每个任务最多处理的样本数")
    args = parser.parse_args()

    print("="*50)
    print(f"开始计算长度为 {args.context_length} 的所有任务的Performer注意力损失")
    print("="*50)

    # 确保输出目录存在
    output_dir = os.path.join(args.output_dir, f"performer_results_{args.context_length}")
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.model_path}")
    config = LlamaConfig.from_pretrained(args.model_path)
    config.verify = True
    
    model = LlamaForCausalLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"模型已加载到设备: {device}")
    
    # 查找指定长度的所有任务
    task_pattern = os.path.join(args.data_dir, f"*_{args.context_length}.json")
    task_files = glob.glob(task_pattern)
    
    ignored_patterns = [
        'cwe',
        'fwe',
        'qa_hotpot',
        'vt',
        'qa_squad'
    ]
    
    task_files = [f for f in task_files if not any(p in f for p in ignored_patterns)]
    
    if not task_files:
        print(f"没有找到长度为 {args.context_length} 的任务文件!")
        return
    
    print(f"找到 {len(task_files)} 个任务文件:")
    for f in task_files:
        print(f"  - {os.path.basename(f)}")
    
    # 存储所有任务的结果
    all_task_results = {}
    
    # 存储各层在所有任务上的平均损失
    aggregated_layer_losses = {}
    
    # 处理每个任务文件
    for task_file in task_files:
        task_name = os.path.basename(task_file).replace(".json", "")
        task_output_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # 处理任务
        task_result = process_task(task_file, model, tokenizer, device, args.max_samples)
        
        # 保存任务结果
        with open(os.path.join(task_output_dir, 'detailed_losses.json'), 'w') as f:
            json.dump(task_result, f, indent=4, ensure_ascii=False)
        
        # 为方便查看，单独保存层平均损失
        with open(os.path.join(task_output_dir, 'layer_average_losses.json'), 'w') as f:
            json.dump(task_result["layer_average_losses"], f, indent=4, ensure_ascii=False)
        
        # 添加到所有任务结果中
        all_task_results[task_name] = task_result
        
        # 聚合到总体层损失
        for layer_idx, loss in task_result["layer_average_losses"].items():
            if layer_idx not in aggregated_layer_losses:
                aggregated_layer_losses[layer_idx] = []
            aggregated_layer_losses[layer_idx].append(loss)
        
        # 尝试绘制图表
        try:
            import matplotlib.pyplot as plt
            
            # 绘制层平均损失
            plt.figure(figsize=(12, 6))
            layers = sorted([int(l) for l in task_result["layer_average_losses"].keys()])
            losses = [task_result["layer_average_losses"][str(layer)] for layer in layers]
            
            plt.plot(layers, losses, marker='o')
            plt.title(f'Mean MSE Loss Across Layers - {task_name}')
            plt.xlabel('Layer Idx')
            plt.ylabel('Mean MSE Loss')
            plt.grid(True)
            plt.savefig(os.path.join(task_output_dir, 'layer_losses.png'))
            plt.close()
            
            # 绘制样本平均损失分布
            if task_result["sample_average_losses"]:
                plt.figure(figsize=(12, 6))
                plt.hist(list(task_result["sample_average_losses"].values()), bins=20)
                plt.title(f'Mean Loss Distribution - {task_name}')
                plt.xlabel('Mean Loss')
                plt.ylabel('Num Samples')
                plt.grid(True)
                plt.savefig(os.path.join(task_output_dir, 'sample_loss_distribution.png'))
                plt.close()
        except ImportError:
            print("未安装 matplotlib，跳过可视化")
        except Exception as e:
            print(f"绘制图表时出错: {e}")

    # 计算每一层在所有任务上的平均、最小、最大、中位数损失
    aggregated_stats = {}
    for layer_idx, losses in aggregated_layer_losses.items():
        aggregated_stats[layer_idx] = {
            "mean": float(np.mean(losses)),
            "min": float(np.min(losses)),
            "max": float(np.max(losses)),
            "median": float(np.median(losses)),
            "std": float(np.std(losses))
        }
    
    # 保存聚合结果
    with open(os.path.join(output_dir, 'aggregated_layer_stats.json'), 'w') as f:
        json.dump(aggregated_stats, f, indent=4, ensure_ascii=False)
    
    # 保存任务概要结果
    task_summaries = {}
    for task_name, result in all_task_results.items():
        task_summaries[task_name] = {
            "overall_avg_loss": result["overall_average_loss"],
            "processed_samples": result["processed_samples"],
            "total_samples": result["total_samples"]
        }
    
    with open(os.path.join(output_dir, 'task_summaries.json'), 'w') as f:
        json.dump(task_summaries, f, indent=4, ensure_ascii=False)
    
    # 绘制聚合结果图表
    try:
        import matplotlib.pyplot as plt
        
        # 绘制所有任务的平均层损失
        plt.figure(figsize=(14, 8))
        layers = sorted([int(l) for l in aggregated_stats.keys()])
        
        # 平均值线
        mean_values = [aggregated_stats[str(l)]["mean"] for l in layers]
        plt.plot(layers, mean_values, 'b-', linewidth=2, label='Mean Loss')
        
        # 填充最大最小区域
        min_values = [aggregated_stats[str(l)]["min"] for l in layers]
        max_values = [aggregated_stats[str(l)]["max"] for l in layers]
        plt.fill_between(layers, min_values, max_values, color='lightblue', alpha=0.3, label='Min-Max Range')
        
        # 标准差区域
        std_values = [aggregated_stats[str(l)]["std"] for l in layers]
        plt.fill_between(layers, 
                         [m - s for m, s in zip(mean_values, std_values)],
                         [m + s for m, s in zip(mean_values, std_values)], 
                         color='blue', alpha=0.2, label='±1 Std Dev')
        
        plt.title(f'Aggregated Layer Losses Across All Tasks - {args.context_length}')
        plt.xlabel('Layer Index')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'aggregated_layer_losses.png'))
        plt.close()
        
        # 绘制任务间的损失对比条形图
        plt.figure(figsize=(16, 8))
        task_names = list(task_summaries.keys())
        task_losses = [task_summaries[task]["overall_avg_loss"] for task in task_names]
        
        # 按损失值排序
        sorted_indices = np.argsort(task_losses)
        sorted_task_names = [task_names[i] for i in sorted_indices]
        sorted_task_losses = [task_losses[i] for i in sorted_indices]
        
        # 绘制条形图
        bars = plt.bar(range(len(sorted_task_names)), sorted_task_losses, color='skyblue')
        plt.xticks(range(len(sorted_task_names)), [n.replace("ruler_", "").replace(f"_{args.context_length}", "") for n in sorted_task_names], rotation=45, ha='right')
        plt.title(f'Overall Loss by Task - {args.context_length}')
        plt.xlabel('Task')
        plt.ylabel('Mean MSE Loss')
        plt.tight_layout()
        plt.grid(axis='y')
        plt.savefig(os.path.join(output_dir, 'task_comparison.png'))
    except ImportError:
        print("未安装 matplotlib，跳过可视化")
    except Exception as e:
        print(f"绘制聚合图表时出错: {e}")
    
    print("="*50)
    print(f"所有任务处理完成! 结果已保存到 {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
