"""
GPT-2 模型架构模块

实现完整的 GPT-2 模型架构，包括：
- LayerNorm: 层归一化
- GELU: GELU 激活函数
- FeedForward: 前馈神经网络
- MultiHeadAttention: 多头注意力机制
- TransformerBlock: Transformer 块
- GPTModel: 完整 GPT 模型
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from config import GPTConfig


#####################################
# 激活函数与归一化
#####################################

class GELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) 激活函数
    
    GELU 是一种平滑的激活函数，近似于 ReLU 但具有更好的数学性质。
    它是 Transformer 模型中的标准激活函数。
    
    数学公式:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    其中：
        - sqrt(2/π) ≈ 0.7978845608
        - 0.044715 是经验常数
    
    形状:
        - Input: (..., d)
        - Output: (..., d)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            GELU 激活后的张量
        """
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)
    
    LayerNorm 对每个样本的所有特征进行归一化，与 BatchNorm 不同，
    它不依赖于批次大小，更适合序列模型和 Transformer 架构。
    
    数学公式:
        1. 计算均值：μ = mean(x, dim=-1)
        2. 计算方差：σ² = var(x, dim=-1, unbiased=False)
        3. 标准化：x_norm = (x - μ) / sqrt(σ² + ε)
        4. 缩放和偏移：output = scale * x_norm + shift
    
    其中：
        - scale 和 shift 是可学习的参数
        - ε 是数值稳定性常数 (1e-5)
    
    形状:
        - Input: (batch, seq_len, emb_dim)
        - Output: (batch, seq_len, emb_dim)
    """
    
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        """
        初始化 LayerNorm
        
        Args:
            emb_dim: 嵌入维度
            eps: 数值稳定性常数，防止除零
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 (初始化为 1)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可学习的偏移参数 (初始化为 0)
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, emb_dim)
            
        Returns:
            归一化后的张量
        """
        # 计算均值和方差（沿最后一个维度）
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch, seq_len, 1)
        
        # 标准化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用可学习的缩放和偏移
        return self.scale * norm_x + self.shift


#####################################
# 前馈神经网络
#####################################

class FeedForward(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)
    
    Transformer 中的前馈网络由两个线性变换和一个激活函数组成：
        FFN(x) = W2 * GELU(W1 * x + b1) + b2
    
    其中中间层的维度通常是嵌入维度的 4 倍，这为模型提供了足够的
    表达能力来学习复杂的特征表示。
    
    结构:
        Linear(emb_dim → 4*emb_dim) → GELU → Linear(4*emb_dim → emb_dim)
    
    形状:
        - Input: (batch, seq_len, emb_dim)
        - Output: (batch, seq_len, emb_dim)
    """
    
    def __init__(self, cfg: GPTConfig):
        """
        初始化前馈网络
        
        Args:
            cfg: GPT 模型配置
        """
        super().__init__()
        self.layers = nn.Sequential(
            # 第一层：扩展到 4 倍维度
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            # GELU 激活函数
            GELU(),
            # 第二层：压缩回原始维度
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, emb_dim)
            
        Returns:
            前馈网络输出
        """
        return self.layers(x)


#####################################
# 多头注意力机制
#####################################

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    多头注意力允许模型同时关注不同位置的不同表示子空间的信息。
    每个头独立计算注意力，然后将结果拼接并通过一个线性变换。
    
    数学公式:
        1. 计算 Query, Key, Value:
           Q = X * W_q, K = X * W_k, V = X * W_v
        2. 分割成多个头:
           Q_i, K_i, V_i (每个头的维度为 head_dim)
        3. 计算缩放点积注意力:
           Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        4. 拼接所有头的输出:
           MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o
    
    其中：
        - head_dim = emb_dim / num_heads
        - 缩放因子 1/sqrt(d_k) 防止点积过大导致梯度消失
        - 因果掩码 (causal mask) 防止未来信息泄露
    
    形状:
        - Input: (batch, seq_len, emb_dim)
        - Output: (batch, seq_len, emb_dim)
    """
    
    def __init__(
        self,
        cfg: GPTConfig,
        attn_type: str = "causal"
    ):
        """
        初始化多头注意力
        
        Args:
            cfg: GPT 模型配置
            attn_type: 注意力类型，'causal' 或 'masked'
        """
        super().__init__()
        
        # 验证输出维度能否被头数整除
        assert cfg.emb_dim % cfg.n_heads == 0, \
            "emb_dim 必须能被 n_heads 整除"
        
        self.cfg = cfg
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.emb_dim // cfg.n_heads  # 每个头的维度
        self.attn_type = attn_type
        
        # 查询、键、值的线性变换
        # d_in = d_out = emb_dim
        self.W_query = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_key = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_value = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        
        # 输出投影层，用于组合多个头的输出
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(cfg.drop_rate)
        
        # 注册因果掩码（上三角矩阵，主对角线以上为 1）
        # 形状：(context_length, context_length)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1)
        )
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将张量分割成多个头
        
        将最后一个维度拆分为 (num_heads, head_dim)，然后转置使头成为第二维，
        便于并行计算所有头的注意力。
        
        Args:
            x: 输入张量 (batch, seq_len, emb_dim)
            
        Returns:
            分割后的张量 (batch, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 重塑：(batch, seq_len, emb_dim) → (batch, seq_len, num_heads, head_dim)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置：(batch, seq_len, num_heads, head_dim) → (batch, num_heads, seq_len, head_dim)
        x = x.transpose(1, 2)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, emb_dim)
            attention_mask: 可选的注意力掩码，用于处理变长序列
            
        Returns:
            注意力输出 (batch, seq_len, emb_dim)
        """
        batch_size, seq_len, d_in = x.shape
        
        # 计算 Query, Key, Value
        # 形状：(batch, seq_len, emb_dim)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # 分割成多个头
        # 形状：(batch, num_heads, seq_len, head_dim)
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        
        # 计算缩放点积注意力
        # attn_scores 形状：(batch, num_heads, seq_len, seq_len)
        attn_scores = queries @ keys.transpose(2, 3)  # 点积
        
        # 缩放因子：1 / sqrt(head_dim)
        # 防止点积过大导致 softmax 梯度消失
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attn_scores = attn_scores * scale_factor
        
        # 应用因果掩码（防止未来信息泄露）
        # 截取到当前序列长度并转换为布尔掩码
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        
        # 使用掩码填充注意力分数（掩码位置设为负无穷）
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # 应用可选的注意力掩码（如 padding mask）
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # 计算注意力权重
        # softmax 沿最后一个维度（key 序列维度）
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 应用 Dropout
        attn_weights = self.dropout(attn_weights)
        
        # 计算上下文向量
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # → (batch, num_heads, seq_len, head_dim)
        context_vec = attn_weights @ values
        
        # 合并头
        # 转置：(batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        
        # 重塑：(batch, seq_len, num_heads, head_dim) → (batch, seq_len, emb_dim)
        context_vec = context_vec.contiguous().view(batch_size, seq_len, self.cfg.emb_dim)
        
        # 输出投影
        context_vec = self.out_proj(context_vec)
        
        return context_vec


#####################################
# Transformer 块
#####################################

class TransformerBlock(nn.Module):
    """
    Transformer 块
    
    每个 Transformer 块包含：
    1. 多头自注意力层（带残差连接和层归一化）
    2. 前馈神经网络（带残差连接和层归一化）
    
    使用 Pre-LayerNorm 架构（先归一化再计算），这种架构更稳定，
    更适合深层模型训练。
    
    计算流程:
        1. 注意力块:
           - 归一化输入
           - 多头注意力
           - Dropout
           - 残差连接
        2. 前馈块:
           - 归一化输入
           - 前馈网络
           - Dropout
           - 残差连接
    
    形状:
        - Input: (batch, seq_len, emb_dim)
        - Output: (batch, seq_len, emb_dim)
    """
    
    def __init__(self, cfg: GPTConfig):
        """
        初始化 Transformer 块
        
        Args:
            cfg: GPT 模型配置
        """
        super().__init__()
        self.cfg = cfg
        
        # 多头注意力层
        self.att = MultiHeadAttention(cfg)
        
        # 前馈神经网络
        self.ff = FeedForward(cfg)
        
        # 两层归一化
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        
        # Dropout 层用于残差连接
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, emb_dim)
            
        Returns:
            Transformer 块输出
        """
        # ========== 注意力块 ==========
        # 保存残差连接
        shortcut = x
        
        # Pre-LayerNorm
        x = self.norm1(x)
        
        # 多头注意力
        # 形状：[batch_size, seq_len, emb_dim]
        x = self.att(x)
        
        # Dropout
        x = self.drop_shortcut(x)
        
        # 残差连接
        x = x + shortcut
        
        # ========== 前馈块 ==========
        # 保存残差连接
        shortcut = x
        
        # Pre-LayerNorm
        x = self.norm2(x)
        
        # 前馈网络
        x = self.ff(x)
        
        # Dropout
        x = self.drop_shortcut(x)
        
        # 残差连接
        x = x + shortcut
        
        return x


#####################################
# GPT 模型
#####################################

class GPTModel(nn.Module):
    """
    GPT-2 模型
    
    完整的 GPT-2 架构，包括：
    1. 词嵌入层（Token Embedding）
    2. 位置嵌入层（Position Embedding）
    3. Transformer 块序列
    4. 最终层归一化
    5. 输出头（词汇表预测）
    
    架构流程:
        输入 → 词嵌入 + 位置嵌入 → Dropout → [TransformerBlock × N] 
             → LayerNorm → 输出头 → Logits
    
    数学公式:
        1. 嵌入层：E_tok = Embedding(vocab_size, emb_dim)
        2. 位置编码：E_pos = Embedding(context_length, emb_dim)
        3. 输入表示：X = E_tok + E_pos
        4. Transformer: H = TransformerBlocks(X)
        5. 输出：Logits = LayerNorm(H) * W_out
    
    形状:
        - Input: (batch, seq_len) - token IDs
        - Output: (batch, seq_len, vocab_size) - logits
    """
    
    def __init__(self, cfg: GPTConfig):
        """
        初始化 GPT 模型
        
        Args:
            cfg: GPT 模型配置
        """
        super().__init__()
        self.cfg = cfg
        
        # ========== 嵌入层 ==========
        # 词嵌入层：将 token ID 映射到嵌入向量
        # 形状：(vocab_size, emb_dim)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        
        # 位置嵌入层：绝对位置编码
        # 形状：(context_length, emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        
        # 嵌入层 Dropout
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        
        # ========== Transformer 块序列 ==========
        # 使用 nn.Sequential 堆叠多个 Transformer 块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        
        # ========== 最终层归一化 ==========
        self.final_norm = LayerNorm(cfg.emb_dim)
        
        # ========== 输出头 ==========
        # 将嵌入维度映射回词汇表大小
        # bias=False 是 GPT-2 的设计选择
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
    
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            in_idx: 输入 token ID 张量 (batch, seq_len)
            
        Returns:
            输出 logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = in_idx.shape
        
        # ========== 词嵌入 ==========
        # 形状：(batch, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        
        # ========== 位置嵌入 ==========
        # 创建位置索引 [0, 1, 2, ..., seq_len-1]
        # 形状：(seq_len,)
        positions = torch.arange(seq_len, device=in_idx.device)
        
        # 位置嵌入
        # 形状：(seq_len, emb_dim)
        pos_embeds = self.pos_emb(positions)
        
        # ========== 合并嵌入 ==========
        # 广播相加：(batch, seq_len, emb_dim) + (seq_len, emb_dim)
        # → (batch, seq_len, emb_dim)
        x = tok_embeds + pos_embeds
        
        # ========== Dropout ==========
        x = self.drop_emb(x)
        
        # ========== Transformer 块序列 ==========
        x = self.trf_blocks(x)
        
        # ========== 最终层归一化 ==========
        x = self.final_norm(x)
        
        # ========== 输出头 ==========
        # 形状：(batch, seq_len, vocab_size)
        logits = self.out_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = False
    ) -> torch.Tensor:
        """
        自回归生成文本
        
        使用自回归方式逐个生成 token，直到达到最大长度。
        支持贪婪解码、temperature 调节和 top-k 采样。
        
        Args:
            idx: 输入上下文 (batch, seq_len)
            max_new_tokens: 生成的最大新 token 数
            temperature: 温度参数，控制生成随机性
                - < 1: 更确定
                - > 1: 更随机
                - = 1: 原始分布
            top_k: Top-k 采样的 k 值，None 表示不使用
            greedy: 是否使用贪婪解码（忽略 temperature 和 top_k）
            
        Returns:
            生成的完整序列 (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # ========== 裁剪上下文 ==========
            # 如果当前上下文超出支持的长度，只保留最后 context_length 个 token
            idx_cond = idx[:, -self.cfg.context_length:]
            
            # ========== 前向传播 ==========
            logits = self(idx_cond)  # (batch, seq_len, vocab_size)
            
            # ========== 获取最后一个时间步的预测 ==========
            # (batch, seq_len, vocab_size) → (batch, vocab_size)
            logits = logits[:, -1, :]
            
            # ========== 应用温度调节 ==========
            # temperature < 1: 分布更尖锐（更确定）
            # temperature > 1: 分布更平坦（更随机）
            if temperature != 1.0:
                logits = logits / temperature
            
            # ========== Top-k 采样 ==========
            if top_k is not None and not greedy:
                # 从词汇表中选择概率最高的 k 个词元进行采样
                # 其余词元的概率被设置为负无穷，从而不会被选中
                # 这种方法可以减少低概率词元对生成结果的影响
                # 同时保持一定的随机性和创造性
                
                # 获取 logits 中值最大的 top_k 个元素及其索引
                top_logits, _ = torch.topk(logits, top_k)
                
                # 获取第 k 大的 logits 值作为阈值
                # 使用 [:, -1, None] 获取每批次中最小的 top_k 值，并保持维度
                min_val = top_logits[:, -1, None]
                
                # 将小于阈值的 logits 设置为负无穷
                # torch.where(condition, x, y) 对满足条件的位置使用 x，否则使用 y
                logits = torch.where(logits < min_val, 
                                     torch.tensor(-float('inf'), device=logits.device), 
                                     logits)
            
            # ========== 采样或贪婪选择 ==========
            if greedy:
                # 贪婪解码策略：在每个时间步选择具有最高概率的词元
                # 这种方法会生成最有可能的序列，但可能导致重复或过于保守的输出
                # torch.argmax 返回指定维度上最大值的索引
                # keepdim=True 保持输出张量的维度与输入相同
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # 随机采样策略：根据 softmax 归一化后的概率分布随机选择词元
                # 这种方法引入随机性，使生成的文本更加多样化和富有创造性
                
                # 将 logits 转换为概率分布，使用 softmax 函数
                # dim=-1 表示在词汇表维度上进行归一化
                probs = torch.softmax(logits, dim=-1)
                
                # 根据概率分布随机采样一个词元
                # num_samples=1 表示每次只采样一个样本
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # ========== 追加到序列 ==========
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def count_parameters(model: nn.Module) -> int:
    """
    统计模型的可训练参数数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # ========== 测试模型 ==========
    from config import GPT_CONFIG_124M, print_config
    
    # 打印配置
    print_config(GPT_CONFIG_124M, "GPT-2 124M")
    
    # 设置随机种子
    torch.manual_seed(123)
    
    # 创建模型
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    
    # 打印参数量
    total_params = count_parameters(model)
    print(f"总参数量：{total_params:,} ({total_params/1e6:.2f}M)")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, GPT_CONFIG_124M.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输入形状：{dummy_input.shape}")
    print(f"输出形状：{output.shape}")
    
    # 测试文本生成
    print("\n" + "="*50)
    print("文本生成测试")
    print("="*50)
    
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    print(f"输入文本：{start_context}")
    print(f"编码后：{encoded}")
    
    # 生成文本
    generated = model.generate(
        idx=encoded_tensor,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50
    )
    
    decoded_text = tokenizer.decode(generated.squeeze(0).tolist())
    print(f"生成文本：{decoded_text}")
