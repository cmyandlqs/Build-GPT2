"""
GPT-2 模型评估脚本

功能：
- 计算验证集 Loss
- 计算困惑度 (Perplexity)
- 生成样本定性评估
- 多指标对比评估

使用示例:
    # 基础评估（Loss + Perplexity）
    python evaluate.py --checkpoint checkpoints/model.pt --data data/val.txt
    
    # 生成样本评估
    python evaluate.py --checkpoint checkpoints/model.pt \\
                       --data data/val.txt \\
                       --metrics loss perplexity generate \\
                       --num_samples 5
    
    # 从 HuggingFace 加载模型评估
    python evaluate.py --hf_model gpt2 --data data/val.txt --metrics perplexity
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import GPTConfig, GPT_CONFIG_124M
from model import GPTModel, count_parameters
from data import create_dataloader_v1, create_instruction_dataloader
from utils import (
    get_device,
    load_weights_from_hf,
    calculate_perplexity,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPT-2 模型评估脚本")
    
    # ========== 模型加载 ==========
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型检查点路径"
    )
    model_group.add_argument(
        "--hf_model",
        type=str,
        default=None,
        help="HuggingFace 模型名称（如 'gpt2'）"
    )
    model_group.add_argument(
        "--ms_model",
        type=str,
        default=None,
        help="ModelScope 模型名称（如 'AI-ModelScope/gpt2'）"
    )

    # ========== 模型来源 ==========
    parser.add_argument(
        "--model_source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="模型权重来源（仅当使用 hf_model 或 ms_model 时有效）"
    )
    
    # ========== 数据参数 ==========
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="评估数据文件路径"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["text", "instruction"],
        default="text",
        help="数据类型"
    )
    
    # ========== 评估参数 ==========
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["loss", "perplexity"],
        choices=["loss", "perplexity", "generate"],
        help="评估指标"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="序列最大长度"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="最大评估批次，None 则评估全部"
    )
    
    # ========== 生成参数 ==========
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="生成样本数量（用于定性评估）"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="生成最大 token 数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="生成温度"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k 采样参数"
    )
    
    # ========== 输出参数 ==========
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细信息"
    )
    
    return parser.parse_args()


def load_model(
    checkpoint_path: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    ms_model_name: Optional[str] = None,
    model_source: str = "huggingface",
    device: torch.device = torch.device("cpu")
) -> Tuple[GPTModel, GPTConfig]:
    """
    加载模型

    Args:
        checkpoint_path: 检查点路径
        hf_model_name: HuggingFace 模型名称
        ms_model_name: ModelScope 模型名称
        model_source: 模型来源（huggingface 或 modelscope）
        device: 计算设备

    Returns:
        (model, config) 元组
    """
    from utils import load_weights_from_hf, load_weights_from_modelscope

    if checkpoint_path is not None:
        print(f"从检查点加载模型：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 获取 model_state_dict（可能在不同层级）
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config_data = checkpoint.get("config", checkpoint.get("args", {}))
        else:
            state_dict = checkpoint
            config_data = checkpoint.get("config", checkpoint.get("args", {}))

        # 从权重中推断实际的 context_length（最可靠）
        if "pos_emb.weight" in state_dict:
            actual_context_length = state_dict["pos_emb.weight"].shape[0]
            print(f"从权重推断 context_length: {actual_context_length}")
        elif "trf_blocks.0.att.mask" in state_dict:
            actual_context_length = state_dict["trf_blocks.0.att.mask"].shape[0]
            print(f"从 attention mask 推断 context_length: {actual_context_length}")
        else:
            # 尝试从配置中获取
            if isinstance(config_data, dict):
                actual_context_length = config_data.get("max_length", config_data.get("context_length", 1024))
            else:
                actual_context_length = 1024
            print(f"从配置文件获取 context_length: {actual_context_length}")

        # 创建配置
        config = GPTConfig(
            vocab_size=50257,
            context_length=actual_context_length,
            emb_dim=GPT_CONFIG_124M.emb_dim,
            n_heads=GPT_CONFIG_124M.n_heads,
            n_layers=GPT_CONFIG_124M.n_layers,
            drop_rate=GPT_CONFIG_124M.drop_rate,
            qkv_bias=GPT_CONFIG_124M.qkv_bias,
        )

        # 创建模型并加载权重
        model = GPTModel(config)
        model.load_state_dict(state_dict)

        print(f"已加载检查点模型 (context_length={actual_context_length})")

    elif ms_model_name is not None:
        print(f"从 ModelScope 加载模型：{ms_model_name}")

        # 根据模型名称选择配置
        if "gpt2-xl" in ms_model_name:
            from config import GPT_CONFIG_1558M
            config = GPT_CONFIG_1558M
        elif "gpt2-large" in ms_model_name:
            from config import GPT_CONFIG_774M
            config = GPT_CONFIG_774M
        elif "gpt2-medium" in ms_model_name:
            from config import GPT_CONFIG_355M
            config = GPT_CONFIG_355M
        else:
            config = GPT_CONFIG_124M

        model = GPTModel(config)
        model = model.to(device)
        model = load_weights_from_modelscope(model, ms_model_name)
        print(f"已加载 ModelScope 预训练模型：{ms_model_name}")

    elif hf_model_name is not None:
        print(f"从 HuggingFace 加载模型：{hf_model_name}")

        # 根据模型名称选择配置
        if "gpt2-xl" in hf_model_name:
            from config import GPT_CONFIG_1558M
            config = GPT_CONFIG_1558M
        elif "gpt2-large" in hf_model_name:
            from config import GPT_CONFIG_774M
            config = GPT_CONFIG_774M
        elif "gpt2-medium" in hf_model_name:
            from config import GPT_CONFIG_355M
            config = GPT_CONFIG_355M
        else:
            config = GPT_CONFIG_124M

        model = GPTModel(config)
        model = model.to(device)

        # 根据来源选择加载方式
        if model_source == "modelscope":
            model = load_weights_from_modelscope(model, hf_model_name)
        else:
            model = load_weights_from_hf(model, hf_model_name)

        print(f"已加载 HuggingFace 预训练模型：{hf_model_name}")
    else:
        raise ValueError("必须指定 checkpoint、hf_model 或 ms_model")

    model = model.to(device)
    model.eval()

    # 打印参数量
    total_params = count_parameters(model)
    print(f"模型参数量：{total_params:,} ({total_params/1e6:.2f}M)")
    
    return model, config


def create_dataloader(
    data_path: str,
    data_type: str,
    batch_size: int,
    max_length: int
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: 数据文件路径
        data_type: 数据类型
        batch_size: 批次大小
        max_length: 序列最大长度
        
    Returns:
        DataLoader
    """
    print(f"\n加载评估数据：{data_path}")
    
    if data_type == "text":
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        dataloader = create_dataloader_v1(
            text,
            tokenizer="tiktoken",
            batch_size=batch_size,
            max_length=max_length,
            stride=max_length,
            shuffle=False,
            drop_last=False
        )
    elif data_type == "instruction":
        dataloader = create_instruction_dataloader(
            data_path=data_path,
            tokenizer="tiktoken",
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
            drop_last=False
        )
    else:
        raise ValueError(f"不支持的数据类型：{data_type}")
    
    print(f"评估批次数量：{len(dataloader)}")
    
    return dataloader


def evaluate_loss_and_perplexity(
    model: GPTModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    verbose: bool = False
) -> Tuple[float, float, List[float]]:
    """
    评估 Loss 和困惑度
    
    Args:
        model: GPT 模型
        dataloader: 数据加载器
        device: 计算设备
        max_batches: 最大评估批次
        verbose: 是否输出详细信息
        
    Returns:
        (avg_loss, perplexity, batch_losses) 元组
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    batch_losses = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    pbar = tqdm(dataloader, desc="评估中", disable=not verbose)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # 解析批次数据
            if isinstance(batch, (tuple, list)):
                x, y = batch
            else:
                x = batch["input_ids"]
                y = batch["labels"]
            
            x = x.to(device)
            y = y.to(device)
            
            # 前向传播
            logits = model(x)
            
            # 计算 loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # 忽略 labels 为 -100 的位置（用于指令微调）
            if isinstance(batch, dict) and "labels" in batch:
                mask = y.view(-1) != -100
                if mask.sum() > 0:
                    loss = loss[mask].mean()
                else:
                    loss = loss.mean()
            else:
                loss = loss.mean()
            
            total_loss += loss.item()
            num_batches += 1
            batch_losses.append(loss.item())
            
            if verbose:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # 计算平均 loss 和困惑度
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity, batch_losses


def evaluate_generation(
    model: GPTModel,
    config: GPTConfig,
    device: torch.device,
    num_samples: int = 3,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50
) -> List[str]:
    """
    评估文本生成质量（定性评估）
    
    Args:
        model: GPT 模型
        config: 模型配置
        device: 计算设备
        num_samples: 生成样本数量
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_k: Top-k 采样参数
        
    Returns:
        生成的文本列表
    """
    import tiktoken
    
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 测试提示
    prompts = [
        "Hello, I am",
        "The future of artificial intelligence",
        "Once upon a time",
        "In a world where",
        "The importance of education",
    ]
    
    generated_texts = []
    
    print("\n" + "="*50)
    print("文本生成定性评估")
    print("="*50)
    
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        
        # 编码
        encoded = tokenizer.encode(prompt)
        encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
        
        # 生成
        with torch.no_grad():
            generated = model.generate(
                idx=encoded_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                greedy=False
            )
        
        # 解码
        generated_ids = generated.squeeze(0).tolist()
        generated_text = tokenizer.decode(generated_ids)
        generated_texts.append(generated_text)
        
        # 输出
        print(f"\n--- 样本 {i+1} ---")
        print(f"提示：{prompt}")
        print(f"生成：{generated_text}")
        print("-" * 50)
    
    return generated_texts


def save_results(
    output_dir: str,
    results: Dict
) -> None:
    """
    保存评估结果
    
    Args:
        output_dir: 输出目录
        results: 评估结果字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存为 JSON
    results_path = output_path / "evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存：{results_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # ========== 获取设备 ==========
    device = get_device()
    
    # ========== 加载模型 ==========
    model, config = load_model(
        checkpoint_path=args.checkpoint,
        hf_model_name=args.hf_model,
        ms_model_name=args.ms_model,
        model_source=args.model_source,
        device=device
    )
    
    # ========== 创建数据加载器 ==========
    dataloader = create_dataloader(
        data_path=args.data,
        data_type=args.data_type,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # ========== 评估 ==========
    results = {
        "model": args.checkpoint or args.hf_model,
        "data": args.data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_batches": args.max_batches,
        }
    }
    
    # Loss 和 Perplexity
    if "loss" in args.metrics or "perplexity" in args.metrics:
        print("\n" + "="*50)
        print("计算 Loss 和 Perplexity")
        print("="*50)
        
        avg_loss, perplexity, batch_losses = evaluate_loss_and_perplexity(
            model=model,
            dataloader=dataloader,
            device=device,
            max_batches=args.max_batches,
            verbose=args.verbose
        )
        
        if "loss" in args.metrics:
            results["loss"] = {
                "average": avg_loss,
                "batch_losses": batch_losses,
            }
            print(f"\n平均 Loss: {avg_loss:.4f}")
        
        if "perplexity" in args.metrics:
            results["perplexity"] = perplexity
            print(f"困惑度 (Perplexity): {perplexity:.2f}")
    
    # 生成评估
    if "generate" in args.metrics:
        generated_texts = evaluate_generation(
            model=model,
            config=config,
            device=device,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        results["generated_samples"] = generated_texts
    
    # ========== 保存结果 ==========
    save_results(args.output_dir, results)
    
    # ========== 打印摘要 ==========
    print("\n" + "="*50)
    print("评估摘要")
    print("="*50)
    
    if "loss" in results:
        print(f"  平均 Loss: {results['loss']['average']:.4f}")
    
    if "perplexity" in results:
        print(f"  困惑度：{results['perplexity']:.2f}")
    
    if "generated_samples" in results:
        print(f"  生成样本数：{len(results['generated_samples'])}")
    
    print("="*50)


if __name__ == "__main__":
    main()
