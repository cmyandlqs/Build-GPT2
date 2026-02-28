"""
工具函数模块

提供训练、评估、推理所需的工具函数：
- Checkpoint 保存与加载
- HuggingFace 权重加载
- 日志记录
- 评估指标计算
"""

import os
import json
import time
import math
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import GPTConfig
from model import GPTModel


#####################################
# Checkpoint 管理
#####################################

class CheckpointManager:
    """
    模型检查点管理器
    
    负责保存和加载训练检查点，包括：
    - 模型权重
    - 优化器状态
    - 训练状态（epoch、step、loss 等）
    - 配置信息
    
    保存格式:
        checkpoint_epoch_00010.pt  # 每 N 个 epoch 保存
        checkpoint_step_001000.pt  # 每 N 个 step 保存
        best_model.pt              # 最佳模型
        latest.pt                  # 最新检查点
    """
    
    def __init__(self, save_dir: str, max_to_keep: int = 3):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 检查点保存目录
            max_to_keep: 最多保留的检查点数量
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.saved_checkpoints = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        is_best: bool = False,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前 epoch
            step: 当前 step
            train_loss: 训练 loss
            val_loss: 验证 loss（可选）
            is_best: 是否是最佳模型
            extra_data: 额外需要保存的数据
            
        Returns:
            保存的文件路径
        """
        # 构建检查点字典
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss
        
        if extra_data is not None:
            checkpoint.update(extra_data)
        
        # 生成文件名
        filename = f"checkpoint_step_{step:06d}.pt"
        filepath = self.save_dir / filename
        
        # 保存
        torch.save(checkpoint, filepath)
        self.saved_checkpoints.append(filepath)
        
        # 管理检查点数量
        if len(self.saved_checkpoints) > self.max_to_keep:
            oldest = self.saved_checkpoints.pop(0)
            if oldest.exists() and not is_best:
                oldest.unlink()
        
        # 保存最新检查点
        latest_path = self.save_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        return str(filepath)
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            model: 模型
            optimizer: 优化器（可选）
            checkpoint_path: 检查点路径，None 则加载最新的
            
        Returns:
            检查点数据字典
        """
        # 确定加载哪个检查点
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / "latest.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError("没有找到检查点文件")
        else:
            checkpoint_path = Path(checkpoint_path)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # 恢复模型权重
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # 恢复优化器状态
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint
    
    def get_latest_path(self) -> Optional[Path]:
        """获取最新检查点路径"""
        latest = self.save_dir / "latest.pt"
        return latest if latest.exists() else None


#####################################
# HuggingFace 权重加载
#####################################

def load_weights_from_hf(
    model: GPTModel,
    model_name: str = "gpt2",
    cache_dir: Optional[str] = None
) -> GPTModel:
    """
    从 HuggingFace 加载预训练的 GPT-2 权重
    
    支持官方 GPT-2 模型：
    - gpt2 (124M)
    - gpt2-medium (355M)
    - gpt2-large (774M)
    - gpt2-xl (1558M)
    
    Args:
        model: 目标 GPT 模型（需要与 model_name 配置匹配）
        model_name: HuggingFace 模型名称
        cache_dir: 缓存目录
        
    Returns:
        加载了预训练权重的模型
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError(
            "需要安装 transformers: pip install transformers"
        )
    
    print(f"正在从 HuggingFace 加载模型：{model_name}")
    
    # 加载 HuggingFace 模型
    hf_model = GPT2LMHeadModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32
    )
    
    # 获取状态字典
    hf_state_dict = hf_model.state_dict()
    
    # 映射 HuggingFace 权重到我们的模型
    state_dict = _map_gpt2_weights(hf_state_dict, model)
    
    # 加载权重
    model.load_state_dict(state_dict, strict=True)
    
    print(f"成功加载 HuggingFace 模型权重：{model_name}")
    
    return model


def load_weights_from_modelscope(
    model: GPTModel,
    model_name: str = "gpt2"
) -> GPTModel:
    """
    从 ModelScope（魔搭）加载预训练的 GPT-2 权重
    
    支持 ModelScope 上的 GPT-2 模型：
    - AI-ModelScope/gpt2 (124M)
    - AI-ModelScope/gpt2-medium (355M)
    - AI-ModelScope/gpt2-large (774M)
    - AI-ModelScope/gpt2-xl (1558M)
    
    Args:
        model: 目标 GPT 模型（需要与 model_name 配置匹配）
        model_name: ModelScope 模型名称
        
    Returns:
        加载了预训练权重的模型
    """
    try:
        from modelscope import snapshot_download
        from transformers import GPT2LMHeadModel
        import os
    except ImportError:
        raise ImportError(
            "需要安装 modelscope 和 transformers: "
            "pip install modelscope transformers"
        )
    
    print(f"正在从 ModelScope 加载模型：{model_name}")
    
    # 下载模型到本地
    model_dir = snapshot_download(model_name)
    
    # 从本地加载模型
    hf_model = GPT2LMHeadModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float32
    )
    
    # 获取状态字典
    hf_state_dict = hf_model.state_dict()
    
    # 映射权重
    state_dict = _map_gpt2_weights(hf_state_dict, model)
    
    # 加载权重
    model.load_state_dict(state_dict, strict=True)
    
    print(f"成功加载 ModelScope 模型权重：{model_name}")
    
    return model


def _map_gpt2_weights(
    hf_state_dict: dict,
    model: GPTModel
) -> dict:
    """
    将 HuggingFace/ModelScope GPT-2 权重映射到我们的模型结构
    
    Args:
        hf_state_dict: HuggingFace 格式的 state_dict
        model: 目标 GPT 模型
        
    Returns:
        映射后的 state_dict
    """
    state_dict = {}
    
    # 词嵌入层
    state_dict["tok_emb.weight"] = hf_state_dict["transformer.wte.weight"]
    
    # 位置嵌入层
    state_dict["pos_emb.weight"] = hf_state_dict["transformer.wpe.weight"]
    
    # Transformer 块
    for i in range(model.cfg.n_layers):
        prefix = f"transformer.h.{i}"
        our_prefix = f"trf_blocks.{i}"
        
        # 层归一化 1
        state_dict[f"{our_prefix}.norm1.scale"] = hf_state_dict[f"{prefix}.ln_1.weight"]
        state_dict[f"{our_prefix}.norm1.shift"] = hf_state_dict[f"{prefix}.ln_1.bias"]
        
        # 层归一化 2
        state_dict[f"{our_prefix}.norm2.scale"] = hf_state_dict[f"{prefix}.ln_2.weight"]
        state_dict[f"{our_prefix}.norm2.shift"] = hf_state_dict[f"{prefix}.ln_2.bias"]
        
        # 注意力层
        # HuggingFace 将 QKV 合并为一个线性层，需要拆分
        c_attn_weight = hf_state_dict[f"{prefix}.attn.c_attn.weight"]
        c_attn_bias = hf_state_dict[f"{prefix}.attn.c_attn.bias"]
        
        embed_dim = model.cfg.emb_dim
        
        # Query
        state_dict[f"{our_prefix}.att.W_query.weight"] = c_attn_weight[:, :embed_dim].t()
        state_dict[f"{our_prefix}.att.W_query.bias"] = c_attn_bias[:embed_dim]
        
        # Key
        state_dict[f"{our_prefix}.att.W_key.weight"] = c_attn_weight[:, embed_dim:embed_dim*2].t()
        state_dict[f"{our_prefix}.att.W_key.bias"] = c_attn_bias[embed_dim:embed_dim*2]
        
        # Value
        state_dict[f"{our_prefix}.att.W_value.weight"] = c_attn_weight[:, embed_dim*2:].t()
        state_dict[f"{our_prefix}.att.W_value.bias"] = c_attn_bias[embed_dim*2:]
        
        # 输出投影
        state_dict[f"{our_prefix}.att.out_proj.weight"] = hf_state_dict[f"{prefix}.attn.c_proj.weight"].t()
        state_dict[f"{our_prefix}.att.out_proj.bias"] = hf_state_dict[f"{prefix}.attn.c_proj.bias"]
        
        # 前馈网络
        state_dict[f"{our_prefix}.ff.layers.0.weight"] = hf_state_dict[f"{prefix}.mlp.c_fc.weight"].t()
        state_dict[f"{our_prefix}.ff.layers.0.bias"] = hf_state_dict[f"{prefix}.mlp.c_fc.bias"]
        state_dict[f"{our_prefix}.ff.layers.2.weight"] = hf_state_dict[f"{prefix}.mlp.c_proj.weight"].t()
        state_dict[f"{our_prefix}.ff.layers.2.bias"] = hf_state_dict[f"{prefix}.mlp.c_proj.bias"]
    
    # 最终层归一化
    state_dict["final_norm.scale"] = hf_state_dict["transformer.ln_f.weight"]
    state_dict["final_norm.shift"] = hf_state_dict["transformer.ln_f.bias"]
    
    # 输出头（与词嵌入共享权重）
    state_dict["out_head.weight"] = hf_state_dict["lm_head.weight"]
    
    return state_dict


def save_to_hf_format(
    model: GPTModel,
    save_dir: str,
    model_name: str = "custom-gpt2"
) -> None:
    """
    将模型保存为 HuggingFace 格式
    
    Args:
        model: GPT 模型
        save_dir: 保存目录
        model_name: 模型名称
    """
    from transformers import GPT2LMHeadModel, GPT2Config
    
    # 创建 HuggingFace 配置
    hf_config = GPT2Config(
        vocab_size=model.cfg.vocab_size,
        n_positions=model.cfg.context_length,
        n_embd=model.cfg.emb_dim,
        n_layer=model.cfg.n_layers,
        n_head=model.cfg.n_heads,
    )
    
    # 创建 HuggingFace 模型
    hf_model = GPT2LMHeadModel(hf_config)
    
    # 权重映射（反向）
    state_dict = {}
    
    # 词嵌入
    state_dict["transformer.wte.weight"] = model.tok_emb.weight.data
    
    # 位置嵌入
    state_dict["transformer.wpe.weight"] = model.pos_emb.weight.data
    
    # Transformer 块
    for i in range(model.cfg.n_layers):
        prefix = f"transformer.h.{i}"
        our_prefix = f"trf_blocks.{i}"
        
        # 层归一化
        state_dict[f"{prefix}.ln_1.weight"] = getattr(model, our_prefix).norm1.scale.data
        state_dict[f"{prefix}.ln_1.bias"] = getattr(model, our_prefix).norm1.shift.data
        state_dict[f"{prefix}.ln_2.weight"] = getattr(model, our_prefix).norm2.scale.data
        state_dict[f"{prefix}.ln_2.bias"] = getattr(model, our_prefix).norm2.shift.data
        
        # 注意力层（需要合并 QKV）
        att = getattr(model, our_prefix).att
        
        # 合并 QKV 权重
        embed_dim = model.cfg.emb_dim
        c_attn_weight = torch.cat([
            att.W_query.weight.data.t(),
            att.W_key.weight.data.t(),
            att.W_value.weight.data.t()
        ], dim=1)
        
        c_attn_bias = torch.cat([
            att.W_query.bias.data,
            att.W_key.bias.data,
            att.W_value.bias.data
        ], dim=0)
        
        # Conv1D 格式
        state_dict[f"{prefix}.attn.c_attn.weight"] = c_attn_weight.t()
        state_dict[f"{prefix}.attn.c_attn.bias"] = c_attn_bias
        
        # 输出投影
        state_dict[f"{prefix}.attn.c_proj.weight"] = att.out_proj.weight.data.t()
        state_dict[f"{prefix}.attn.c_proj.bias"] = att.out_proj.bias.data
        
        # 前馈网络
        state_dict[f"{prefix}.mlp.c_fc.weight"] = att.ff.layers[0].weight.data.t()
        state_dict[f"{prefix}.mlp.c_fc.bias"] = att.ff.layers[0].bias.data
        state_dict[f"{prefix}.mlp.c_proj.weight"] = att.ff.layers[2].weight.data.t()
        state_dict[f"{prefix}.mlp.c_proj.bias"] = att.ff.layers[2].bias.data
    
    # 最终层归一化
    state_dict["transformer.ln_f.weight"] = model.final_norm.scale.data
    state_dict["transformer.ln_f.bias"] = model.final_norm.shift.data
    
    # 输出头
    state_dict["lm_head.weight"] = model.out_head.weight.data
    
    # 加载权重
    hf_model.load_state_dict(state_dict)
    
    # 保存
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    hf_model.save_pretrained(str(save_path))
    
    print(f"模型已保存到 HuggingFace 格式：{save_path}")


#####################################
# 日志记录
#####################################

class TrainingLogger:
    """
    训练日志记录器
    
    支持：
    - TensorBoard 日志
    - 命令行实时输出
    - JSON 日志文件
    """
    
    def __init__(
        self,
        log_dir: str,
        model_config: Optional[GPTConfig] = None,
        console_output: bool = True
    ):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            model_config: 模型配置（可选）
            console_output: 是否输出到控制台
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console_output = console_output
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # 日志数据
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # 记录配置
        if model_config is not None:
            self._log_config(model_config)
    
    def _log_config(self, config: GPTConfig) -> None:
        """记录模型配置到 TensorBoard"""
        config_dict = {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "emb_dim": config.emb_dim,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "drop_rate": config.drop_rate,
        }
        
        # 添加到 TensorBoard hparams
        self.writer.add_hparams(
            {k: str(v) for k, v in config_dict.items()},
            {"hparam/dummy": 0}
        )
    
    def log_train_step(
        self,
        step: int,
        loss: float,
        lr: float,
        epoch: Optional[int] = None
    ) -> None:
        """
        记录训练步
        
        Args:
            step: 全局 step
            loss: loss 值
            lr: 学习率
            epoch: 当前 epoch（可选）
        """
        # TensorBoard
        self.writer.add_scalar("Loss/Train", loss, step)
        self.writer.add_scalar("LearningRate", lr, step)
        
        if epoch is not None:
            self.writer.add_scalar("Epoch", epoch, step)
        
        # 记录数据
        self.train_losses.append({"step": step, "epoch": epoch, "loss": loss})
        self.learning_rates.append({"step": step, "lr": lr})
        
        # 控制台输出
        if self.console_output:
            epoch_str = f"Epoch {epoch} | " if epoch is not None else ""
            print(f"{epoch_str}Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e}")
    
    def log_validation(
        self,
        step: int,
        val_loss: float,
        perplexity: Optional[float] = None
    ) -> None:
        """
        记录验证结果
        
        Args:
            step: 全局 step
            val_loss: 验证 loss
            perplexity: 困惑度（可选）
        """
        # TensorBoard
        self.writer.add_scalar("Loss/Validation", val_loss, step)
        
        if perplexity is not None:
            self.writer.add_scalar("Perplexity", perplexity, step)
        
        # 记录数据
        self.val_losses.append({"step": step, "loss": val_loss})
    
    def save_logs(self) -> None:
        """保存日志到 JSON 文件"""
        logs = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates
        }
        
        log_path = self.log_dir / "training_logs.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
        
        print(f"训练日志已保存到：{log_path}")
    
    def close(self) -> None:
        """关闭日志记录器"""
        self.writer.close()
        self.save_logs()


#####################################
# 评估指标
#####################################

def calculate_perplexity(loss: float) -> float:
    """
    计算困惑度 (Perplexity)
    
    困惑度是语言模型的常用评估指标，表示模型对下一个 token 预测的不确定性。
    困惑度越低，模型性能越好。
    
    数学公式:
        Perplexity = exp(CrossEntropyLoss)
    
    Args:
        loss: 交叉熵损失值
        
    Returns:
        困惑度
    """
    return math.exp(loss)


def evaluate_model(
    model: GPTModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Tuple[float, float]:
    """
    评估模型
    
    在验证集上计算平均 loss 和困惑度。
    
    Args:
        model: GPT 模型
        dataloader: 数据加载器
        device: 计算设备
        max_batches: 最大评估批次，None 则评估全部
        
    Returns:
        (avg_loss, perplexity) 元组
    """
    import math
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # 处理不同的数据格式
            if isinstance(batch, (tuple, list)):
                x, y = batch
                attention_mask = None
            else:
                x = batch["input_ids"]
                y = batch["labels"]
                attention_mask = batch.get("attention_mask")
            
            # 移动到设备
            x = x.to(device)
            y = y.to(device)
            
            # 前向传播
            logits = model(x)
            
            # 计算 loss
            # 调整形状：(batch, seq_len, vocab_size) → (batch*seq_len, vocab_size)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    
    # 计算平均 loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # 计算困惑度
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


#####################################
# 学习率调度器
#####################################

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    创建带预热的余弦学习率调度器
    
    学习率变化曲线：
    1. 预热阶段：从 0 线性增长到初始学习率
    2. 衰减阶段：按余弦函数衰减到最小学习率
    
    公式:
        warmup: lr = initial_lr * (step / warmup_steps)
        decay: lr = initial_lr * 0.5 * (1 + cos(π * (step - warmup) / (total - warmup)))
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率与初始学习率的比值
        
    Returns:
        LambdaLR 调度器
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int) -> float:
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 余弦衰减阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


#####################################
# 其他工具函数
#####################################

def count_parameters(model: nn.Module) -> int:
    """
    统计模型的可训练参数数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    格式化时间为可读字符串
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串 (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    elif minutes > 0:
        return f"{minutes:02d}:{secs:02d}"
    else:
        return f"{secs:.1f}s"


def get_device() -> torch.device:
    """
    获取可用的计算设备
    
    优先级：CUDA > MPS (macOS) > CPU
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    
    return device


def set_seed(seed: int) -> None:
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 导入 math 模块（用于 perplexity 计算）
import math
