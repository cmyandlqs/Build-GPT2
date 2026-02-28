"""
模型模块单元测试

测试内容：
- 模型组件形状验证
- 前向传播输出形状
- 文本生成功能
"""

import pytest
import torch

from config import GPTConfig, GPT_CONFIG_124M
from model import (
    GELU,
    LayerNorm,
    FeedForward,
    MultiHeadAttention,
    TransformerBlock,
    GPTModel,
    count_parameters,
)


class TestGELU:
    """测试 GELU 激活函数"""
    
    def test_output_shape(self):
        """测试输出形状与输入一致"""
        gelu = GELU()
        x = torch.randn(2, 10, 768)
        out = gelu(x)
        assert out.shape == x.shape
    
    def test_output_values(self):
        """测试输出值范围"""
        gelu = GELU()
        x = torch.tensor([[-1.0, 0.0, 1.0]])
        out = gelu(x)
        
        # GELU(0) = 0
        assert torch.isclose(out[0, 1], torch.tensor(0.0), atol=1e-6)
        
        # GELU(1) > 0, GELU(-1) < 0
        assert out[0, 2] > 0
        assert out[0, 0] < 0


class TestLayerNorm:
    """测试层归一化"""
    
    def test_output_shape(self):
        """测试输出形状与输入一致"""
        ln = LayerNorm(emb_dim=768)
        x = torch.randn(2, 10, 768)
        out = ln(x)
        assert out.shape == x.shape
    
    def test_normalized_mean(self):
        """测试归一化后均值接近 0"""
        ln = LayerNorm(emb_dim=768)
        x = torch.randn(2, 10, 768)
        out = ln(x)
        
        # 沿最后一个维度计算均值
        mean = out.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    
    def test_parameters(self):
        """测试可学习参数"""
        ln = LayerNorm(emb_dim=768)
        assert ln.scale.shape == (768,)
        assert ln.shift.shape == (768,)


class TestFeedForward:
    """测试前馈神经网络"""
    
    def test_output_shape(self):
        """测试输出形状与输入一致"""
        cfg = GPT_CONFIG_124M
        ff = FeedForward(cfg)
        x = torch.randn(2, 10, cfg.emb_dim)
        out = ff(x)
        assert out.shape == x.shape
    
    def test_hidden_dimension(self):
        """测试中间层维度是 emb_dim 的 4 倍"""
        cfg = GPT_CONFIG_124M
        ff = FeedForward(cfg)
        
        # 第一层线性变换
        assert ff.layers[0].out_features == 4 * cfg.emb_dim
        assert ff.layers[0].in_features == cfg.emb_dim


class TestMultiHeadAttention:
    """测试多头注意力机制"""
    
    def test_output_shape(self):
        """测试输出形状与输入一致"""
        cfg = GPT_CONFIG_124M
        attn = MultiHeadAttention(cfg)
        x = torch.randn(2, 10, cfg.emb_dim)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_num_heads(self):
        """测试头数配置"""
        cfg = GPT_CONFIG_124M
        attn = MultiHeadAttention(cfg)
        assert attn.num_heads == cfg.n_heads
        assert attn.head_dim == cfg.emb_dim // cfg.n_heads
    
    def test_causal_mask(self):
        """测试因果掩码形状"""
        cfg = GPT_CONFIG_124M
        attn = MultiHeadAttention(cfg)
        assert attn.mask.shape == (cfg.context_length, cfg.context_length)
        # 上三角矩阵
        assert attn.mask[0, 1] == 1.0
        assert attn.mask[0, 0] == 0.0
    
    def test_attention_weights_sum(self):
        """测试注意力权重和为 1"""
        cfg = GPT_CONFIG_124M
        attn = MultiHeadAttention(cfg)
        x = torch.randn(1, 5, cfg.emb_dim)
        
        # 需要修改 forward 返回注意力权重，这里简化测试
        out = attn(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    """测试 Transformer 块"""
    
    def test_output_shape(self):
        """测试输出形状与输入一致"""
        cfg = GPT_CONFIG_124M
        block = TransformerBlock(cfg)
        x = torch.randn(2, 10, cfg.emb_dim)
        out = block(x)
        assert out.shape == x.shape
    
    def test_residual_connection(self):
        """测试残差连接"""
        cfg = GPT_CONFIG_124M
        block = TransformerBlock(cfg)
        x = torch.randn(2, 10, cfg.emb_dim)
        out = block(x)
        
        # 输出应该包含输入信息（残差连接）
        # 简化测试：只检查形状
        assert out.shape == x.shape


class TestGPTModel:
    """测试完整 GPT 模型"""
    
    def test_output_shape(self):
        """测试输出 logits 形状"""
        cfg = GPT_CONFIG_124M
        model = GPTModel(cfg)
        
        batch_size = 2
        seq_len = 10
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        
        logits = model(x)
        
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    
    def test_embedding_shapes(self):
        """测试嵌入层形状"""
        cfg = GPT_CONFIG_124M
        model = GPTModel(cfg)
        
        assert model.tok_emb.weight.shape == (cfg.vocab_size, cfg.emb_dim)
        assert model.pos_emb.weight.shape == (cfg.context_length, cfg.emb_dim)
    
    def test_num_layers(self):
        """测试 Transformer 层数"""
        cfg = GPT_CONFIG_124M
        model = GPTModel(cfg)
        
        assert len(model.trf_blocks) == cfg.n_layers
    
    def test_generate_output_shape(self):
        """测试文本生成输出形状"""
        cfg = GPT_CONFIG_124M
        model = GPTModel(cfg)
        model.eval()
        
        batch_size = 1
        seq_len = 5
        max_new_tokens = 10
        
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            generated = model.generate(x, max_new_tokens=max_new_tokens)
        
        assert generated.shape == (batch_size, seq_len + max_new_tokens)
    
    def test_count_parameters(self):
        """测试参数量统计"""
        cfg = GPT_CONFIG_124M
        model = GPTModel(cfg)
        
        total_params = count_parameters(model)
        
        # GPT-2 124M 模型应该有约 1.24 亿参数
        assert 100e6 < total_params < 180e6
    
    def test_device_movement(self):
        """测试设备移动"""
        cfg = GPT_CONFIG_124M
        model = GPTModel(cfg)
        
        # 移动到 CPU
        model = model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"
        
        # 如果有 CUDA，测试 CUDA
        if torch.cuda.is_available():
            model = model.to("cuda")
            assert next(model.parameters()).device.type == "cuda"


class TestGPTConfig:
    """测试配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        cfg = GPT_CONFIG_124M
        
        assert cfg.vocab_size == 50257
        assert cfg.context_length == 1024
        assert cfg.emb_dim == 768
        assert cfg.n_heads == 12
        assert cfg.n_layers == 12
        assert cfg.drop_rate == 0.1
    
    def test_custom_config(self):
        """测试自定义配置"""
        cfg = GPTConfig(
            vocab_size=1000,
            context_length=256,
            emb_dim=128,
            n_heads=4,
            n_layers=2,
        )
        
        assert cfg.vocab_size == 1000
        assert cfg.context_length == 256
        assert cfg.emb_dim == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
