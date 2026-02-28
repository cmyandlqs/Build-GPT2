"""
Pytest 配置文件

设置共享的 pytest fixture 和配置
"""

import pytest
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """pytest 配置钩子"""
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """获取可用的计算设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def small_config():
    """小型模型配置（用于快速测试）"""
    from config import GPTConfig
    
    return GPTConfig(
        vocab_size=50257,
        context_length=64,
        emb_dim=128,
        n_heads=4,
        n_layers=2,
        drop_rate=0.0,  # 测试时禁用 dropout
        qkv_bias=False,
    )


@pytest.fixture
def tiny_config():
    """微型模型配置（用于最快速测试）"""
    from config import GPTConfig
    
    return GPTConfig(
        vocab_size=1000,
        context_length=32,
        emb_dim=64,
        n_heads=2,
        n_layers=1,
        drop_rate=0.0,
        qkv_bias=False,
    )


@pytest.fixture
def sample_text():
    """示例文本数据"""
    return "This is a sample text for testing. " * 100


@pytest.fixture
def temp_dir(tmp_path):
    """临时目录 fixture"""
    return str(tmp_path)
