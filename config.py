"""
模型配置模块

定义 GPT-2 模型的各种配置参数，支持不同规模的模型。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """
    GPT 模型配置类
    
    使用 dataclass 提供简洁的配置定义，支持类型检查和默认值。
    
    Attributes:
        vocab_size: 词汇表大小，GPT-2 默认为 50257
        context_length: 上下文长度，即最大支持的 token 序列长度
        emb_dim: 嵌入层维度，即模型的隐藏层维度
        n_heads: 多头注意力的头数
        n_layers: Transformer 块的数量
        drop_rate: Dropout 比率，用于防止过拟合
        qkv_bias: Query、Key、Value 线性变换是否使用偏置
        pos_encoding_type: 位置编码类型，支持 'absolute' 或 'rope'
    """
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    pos_encoding_type: str = "absolute"  # 'absolute' 或 'rope'
    

# 预定义的模型配置
# GPT-2 官方配置参考：https://github.com/openai/gpt-2/blob/master/src/hparams.py
# 注意: qkv_bias=True 以匹配官方 GPT-2（使用 c_attn.bias）
GPT_CONFIG_124M = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=True,  # 官方 GPT-2 的 attention 层有 bias
)

GPT_CONFIG_355M = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1024,
    n_heads=16,
    n_layers=24,
    drop_rate=0.1,
    qkv_bias=True,
)

GPT_CONFIG_774M = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1280,
    n_heads=20,
    n_layers=36,
    drop_rate=0.1,
    qkv_bias=True,
)

GPT_CONFIG_1558M = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1600,
    n_heads=25,
    n_layers=48,
    drop_rate=0.1,
    qkv_bias=True,
)

# 配置名称到配置对象的映射，便于命令行参数使用
MODEL_CONFIGS = {
    "124M": GPT_CONFIG_124M,
    "355M": GPT_CONFIG_355M,
    "774M": GPT_CONFIG_774M,
    "1558M": GPT_CONFIG_1558M,
}


def get_model_config(model_name: str) -> GPTConfig:
    """
    根据模型名称获取配置
    
    Args:
        model_name: 模型名称，如 '124M', '355M' 等
        
    Returns:
        对应的 GPTConfig 配置对象
        
    Raises:
        ValueError: 如果模型名称不存在
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"未知的模型配置：{model_name}，"
            f"可选的模型配置：{list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]


def print_config(config: GPTConfig, model_name: str = "Custom") -> None:
    """
    打印模型配置信息
    
    Args:
        config: GPTConfig 配置对象
        model_name: 模型名称标签
    """
    # 计算模型参数量（近似值）
    # 公式参考：https://arxiv.org/abs/2001.08361
    # 参数量 ≈ 12 * n_layers * emb_dim^2 (主要部分)
    approx_params = 12 * config.n_layers * (config.emb_dim ** 2)
    
    print(f"\n{'='*50}")
    print(f"模型配置：{model_name}")
    print(f"{'='*50}")
    print(f"  词汇表大小 (vocab_size):    {config.vocab_size:,}")
    print(f"  上下文长度 (context_length): {config.context_length:,}")
    print(f"  嵌入维度 (emb_dim):         {config.emb_dim:,}")
    print(f"  注意力头数 (n_heads):       {config.n_heads}")
    print(f"  Transformer 层数 (n_layers): {config.n_layers}")
    print(f"  Dropout 率 (drop_rate):     {config.drop_rate}")
    print(f"  QKV 偏置 (qkv_bias):         {config.qkv_bias}")
    print(f"  位置编码类型 (pos_encoding): {config.pos_encoding_type}")
    print(f"  近似参数量：                {approx_params/1e6:.2f}M")
    print(f"{'='*50}\n")
