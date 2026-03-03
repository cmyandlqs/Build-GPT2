"""
GPT-2 文本生成脚本

功能：
- 从检查点加载模型
- 从 HuggingFace 加载模型
- 支持多种解码策略（贪婪、Top-k、Temperature）
- 交互式生成和批量生成

使用示例:
    # 从检查点生成
    python generate.py --checkpoint checkpoints/model.pt --prompt "Hello, I am"
    
    # 从 HuggingFace 加载
    python generate.py --hf_model gpt2 --prompt "Once upon a time"
    
    # Top-k + Temperature 采样
    python generate.py --checkpoint checkpoints/model.pt \
                       --prompt "The future of AI" \
                       --top_k 50 --temperature 0.8
    
    # 交互式模式
    python generate.py --checkpoint checkpoints/model.pt --interactive
"""

import argparse
import sys
import torch
from typing import Optional, List

from config import GPT_CONFIG_124M, GPTConfig
from model import GPTModel
from utils import get_device, load_weights_from_hf


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPT-2 文本生成脚本")
    
    # ========== 模型加载 ==========
    model_group = parser.add_mutually_exclusive_group(required=False)
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
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="使用随机初始化的模型（不加载任何权重）"
    )
    
    # ========== 模型来源 ==========
    parser.add_argument(
        "--model_source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="模型权重来源（仅当使用 hf_model 或 ms_model 时有效）"
    )
    
    # ========== 生成参数 ==========
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, I am",
        help="生成提示文本"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="温度参数（>1 更随机，<1 更确定）"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k 采样的 k 值，None 表示不使用"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="使用贪婪解码（忽略 temperature 和 top_k）"
    )
    
    # ========== 模式 ==========
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式（多轮对话）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="生成样本数量"
    )
    
    # ========== 其他参数 ==========
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=1024,
        help="上下文长度"
    )
    
    return parser.parse_args()


def load_model(
    checkpoint_path: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    ms_model_name: Optional[str] = None,
    model_source: str = "huggingface",
    device: torch.device = torch.device("cpu"),
    random_init: bool = False
) -> tuple[GPTModel, GPTConfig]:
    """
    加载模型

    Args:
        checkpoint_path: 检查点路径
        hf_model_name: HuggingFace 模型名称
        ms_model_name: ModelScope 模型名称
        model_source: 模型来源（huggingface 或 modelscope）
        device: 计算设备
        random_init: 是否使用随机初始化

    Returns:
        (model, config) 元组
    """
    from utils import load_weights_from_hf, load_weights_from_modelscope

    # 随机初始化
    if random_init:
        print("使用随机初始化的模型")
        config = GPT_CONFIG_124M
        model = GPTModel(config)
        print(f"已创建随机模型 (context_length={config.context_length})")

    elif checkpoint_path is not None:
        # 从检查点加载
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

        # 创建模型
        model = GPTModel(config)

        # 加载权重
        model.load_state_dict(state_dict)

        print(f"已加载检查点模型 (context_length={actual_context_length})")

    elif ms_model_name is not None:
        # 从 ModelScope 加载
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

        # 创建模型
        model = GPTModel(config)
        model = model.to(device)

        # 加载 ModelScope 权重
        model = load_weights_from_modelscope(model, ms_model_name)

    elif hf_model_name is not None:
        # 从 HuggingFace 加载
        print(f"从 HuggingFace 加载模型：{hf_model_name}")

        # 根据模型名称选择配置
        if hf_model_name == "gpt2":
            config = GPT_CONFIG_124M
        elif hf_model_name == "gpt2-medium":
            from config import GPT_CONFIG_355M
            config = GPT_CONFIG_355M
        elif hf_model_name == "gpt2-large":
            from config import GPT_CONFIG_774M
            config = GPT_CONFIG_774M
        elif hf_model_name == "gpt2-xl":
            from config import GPT_CONFIG_1558M
            config = GPT_CONFIG_1558M
        else:
            config = GPT_CONFIG_124M

        # 创建模型
        model = GPTModel(config)
        model = model.to(device)

        # 根据来源选择加载方式
        if model_source == "modelscope":
            model = load_weights_from_modelscope(model, hf_model_name)
        else:
            model = load_weights_from_hf(model, hf_model_name)

    model = model.to(device)
    model.eval()

    return model, config


def generate_text(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    greedy: bool,
    device: torch.device
) -> str:
    """
    生成文本
    
    Args:
        model: GPT 模型
        tokenizer: 分词器
        prompt: 提示文本
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_k: Top-k 采样参数
        greedy: 是否贪婪解码
        device: 计算设备
        
    Returns:
        生成的文本
    """
    # 编码提示
    encoded_prompt = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"输入：{prompt}")
    print(f"编码长度：{len(encoded_prompt)}")
    
    # 生成
    with torch.no_grad():
        generated = model.generate(
            idx=encoded_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy
        )
    
    # 解码
    generated_ids = generated.squeeze(0).tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


def interactive_mode(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace
) -> None:
    """
    交互式生成模式
    
    Args:
        model: GPT 模型
        tokenizer: 分词器
        device: 计算设备
        args: 命令行参数
    """
    print("\n" + "="*50)
    print("交互式生成模式")
    print("输入 'quit' 或 'exit' 退出")
    print("="*50 + "\n")
    
    history = ""
    
    while True:
        try:
            # 获取用户输入
            user_input = input(">>> ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见!")
                break
            
            if not user_input:
                continue
            
            # 构建提示
            if history:
                prompt = history + "\n" + user_input
            else:
                prompt = user_input
            
            # 生成
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                greedy=args.greedy,
                device=device
            )
            
            print(f"\n生成：{response}\n")
            
            # 更新历史（可选）
            # history = response
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误：{e}")


def main():
    """主函数"""
    args = parse_args()
    
    # ========== 设置设备 ==========
    device = get_device()

    # 检查是否指定了模型来源
    if not any([args.checkpoint, args.hf_model, args.ms_model, args.random_init]):
        raise ValueError("必须指定 --checkpoint、--hf_model、--ms_model 或 --random_init 之一")

    # ========== 加载模型 ==========
    model, config = load_model(
        checkpoint_path=args.checkpoint,
        hf_model_name=args.hf_model,
        ms_model_name=args.ms_model,
        model_source=args.model_source,
        device=device,
        random_init=args.random_init
    )
    
    # ========== 加载分词器 ==========
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # ========== 打印生成配置 ==========
    print("\n" + "="*50)
    print("生成配置")
    print("="*50)
    print(f"提示文本：{args.prompt}")
    print(f"最大新 token 数：{args.max_new_tokens}")
    print(f"温度：{args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"贪婪解码：{args.greedy}")
    print(f"样本数：{args.num_samples}")
    print("="*50 + "\n")
    
    # ========== 交互式模式 ==========
    if args.interactive:
        interactive_mode(model, tokenizer, device, args)
        return
    
    # ========== 批量生成 ==========
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n{'='*50}")
            print(f"样本 {i+1}/{args.num_samples}")
            print(f"{'='*50}\n")
        
        # 生成文本
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
            device=device
        )
        
        print(f"\n{'='*50}")
        print("生成结果")
        print(f"{'='*50}")
        print(generated_text)
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
