"""
GPT-2 预训练脚本

功能：
- 从文本文件加载数据
- 训练 GPT-2 模型
- 支持断点续训
- TensorBoard 日志记录
- 定期保存检查点

使用示例:
    # 基础训练
    python train.py --data data/the-verdict.txt --epochs 10 --batch_size 1
    python train.py --data data/the-verdict.txt --epochs 10 --batch_size 1 --max_length 16 --mixed_precision
    
    # 从检查点恢复
    python train.py --data data/the-verdict.txt --resume checkpoints/latest.pt
    
    # 使用 HuggingFace 预训练权重
    python train.py --data data/the-verdict.txt --hf_model gpt2 --epochs 5
"""

import argparse
import os
import gc
import time
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import GPTConfig, GPT_CONFIG_124M, get_model_config, print_config
from model import GPTModel, count_parameters
from data import create_dataloader_v1
from utils import (
    CheckpointManager,
    TrainingLogger,
    evaluate_model,
    get_cosine_schedule_with_warmup,
    get_device,
    set_seed,
    format_time,
)


def clear_gpu_memory():
    """
    清理 GPU 显存
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPT-2 预训练脚本")
    
    # ========== 数据参数 ==========
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="训练数据文件路径（txt 格式）"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="验证数据文件路径（可选）"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["tiktoken", "bpe"],
        default="tiktoken",
        help="分词器类型"
    )
    
    # ========== 模型参数 ==========
    parser.add_argument(
        "--model_config",
        type=str,
        default="124M",
        choices=["124M", "355M", "774M", "1558M"],
        help="模型配置"
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default=None,
        help="从 HuggingFace 加载预训练模型（如 'gpt2'）"
    )
    parser.add_argument(
        "--ms_model",
        type=str,
        default=None,
        help="从 ModelScope 加载预训练模型（如 'AI-ModelScope/gpt2'）"
    )
    parser.add_argument(
        "--model_source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="模型权重来源"
    )
    
    # ========== 训练参数 ==========
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数"
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
        "--stride",
        type=int,
        default=128,
        help="滑动窗口步长"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="学习率"
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="最小学习率比例"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="预热阶段比例"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="权重衰减"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="梯度裁剪阈值"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数"
    )
    
    # ========== 检查点和日志 ==========
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="输出目录"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="TensorBoard 日志目录"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100,
        help="评估频率（每 N 步）"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=1000,
        help="保存检查点频率（每 N 步）"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )
    
    # ========== 其他参数 ==========
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="数据加载进程数"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="使用混合精度训练"
    )
    
    return parser.parse_args()


def create_model(
    config: GPTConfig,
    hf_model_name: Optional[str] = None,
    ms_model_name: Optional[str] = None,
    model_source: str = "huggingface",
    device: torch.device = torch.device("cpu")
) -> GPTModel:
    """
    创建模型

    Args:
        config: 模型配置
        hf_model_name: HuggingFace 模型名称（可选）
        ms_model_name: ModelScope 模型名称（可选）
        model_source: 模型来源（huggingface 或 modelscope）
        device: 计算设备

    Returns:
        GPT 模型
    """
    from utils import load_weights_from_hf, load_weights_from_modelscope

    # 创建模型
    model = GPTModel(config)

    # 打印配置
    model_name = hf_model_name or ms_model_name or f"GPT-{config.emb_dim}"
    print_config(config, model_name)

    # 打印参数量
    total_params = count_parameters(model)
    print(f"总参数量：{total_params:,} ({total_params/1e6:.2f}M)")

    # 移动到设备
    model = model.to(device)

    # 加载 HuggingFace 权重
    if hf_model_name is not None:
        if model_source == "modelscope":
            model = load_weights_from_modelscope(model, hf_model_name)
        else:
            model = load_weights_from_hf(model, hf_model_name)
        print(f"已加载预训练权重：{hf_model_name} ({model_source})")

    # 加载 ModelScope 权重
    elif ms_model_name is not None:
        model = load_weights_from_modelscope(model, ms_model_name)
        print(f"已加载 ModelScope 预训练权重：{ms_model_name}")

    return model


def create_optimizer_and_scheduler(
    model: GPTModel,
    learning_rate: float,
    min_lr_ratio: float,
    warmup_ratio: float,
    num_training_steps: int,
    weight_decay: float = 0.01
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    创建优化器和学习率调度器
    
    Args:
        model: GPT 模型
        learning_rate: 初始学习率
        min_lr_ratio: 最小学习率比例
        warmup_ratio: 预热比例
        num_training_steps: 总训练步数
        weight_decay: 权重衰减
        
    Returns:
        (optimizer, scheduler) 元组
    """
    # 创建 AdamW 优化器
    # 分离偏置和 LayerNorm 参数，不应用权重衰减
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "ln_1", "ln_2", "norm1", "norm2", "final_norm"]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # 计算预热步数
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    # 创建学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio
    )
    
    return optimizer, scheduler


def train_epoch(
    model: GPTModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
    val_dataloader: Optional[DataLoader] = None,
    use_amp: bool = False
) -> Tuple[float, int]:
    """
    训练一个 epoch
    
    Args:
        model: GPT 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        device: 计算设备
        epoch: 当前 epoch
        global_step: 全局 step
        args: 命令行参数
        logger: 日志记录器
        checkpoint_manager: 检查点管理器
        val_dataloader: 验证数据加载器（可选）
        use_amp: 是否使用混合精度
        
    Returns:
        (avg_loss, global_step) 元组
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 混合精度缩放器
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    
    # 进度条
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch+1}/{args.epochs}",
        leave=True
    )
    
    for batch_idx, batch in enumerate(pbar):
        # 解析批次数据
        if isinstance(batch, (tuple, list)):
            x, y = batch
        else:
            x = batch["input_ids"]
            y = batch["labels"]
        
        x = x.to(device)
        y = y.to(device)
        
        # 自动混合精度
        if use_amp:
            with torch.amp.autocast(device.type):
                logits = model(x)
                
                # 计算 loss
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
                
                # 梯度累积
                loss = loss / args.gradient_accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 更新权重
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip
                )
                
                # 优化器 step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            # 标准训练
            logits = model(x)
            
            # 计算 loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # 梯度累积
            loss = loss / args.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 更新权重
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip
                )
                
                # 优化器 step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # 定期清理显存（每 100 步）
        if global_step % 100 == 0 and device.type == "cuda":
            clear_gpu_memory()

        # 记录 loss
        total_loss += loss.item() * args.gradient_accumulation_steps
        num_batches += 1
        global_step += 1
        
        # 获取当前学习率
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录日志
        if global_step % 10 == 0:
            logger.log_train_step(
                step=global_step,
                loss=loss.item() * args.gradient_accumulation_steps,
                lr=current_lr,
                epoch=epoch + 1
            )
        
        # 验证
        if val_dataloader is not None and global_step % args.eval_freq == 0:
            val_loss, perplexity = evaluate_model(
                model,
                val_dataloader,
                device
            )
            logger.log_validation(global_step, val_loss, perplexity)
            
            pbar.set_postfix({
                "train_loss": f"{total_loss / num_batches:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "ppl": f"{perplexity:.1f}"
            })
        
        # 保存检查点
        if global_step % args.checkpoint_freq == 0:
            avg_train_loss = total_loss / num_batches
            val_loss = None
            if val_dataloader is not None:
                val_loss, _ = evaluate_model(model, val_dataloader, device)
            
            checkpoint_path = checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                extra_data={
                    "args": vars(args),
                    "scheduler_state_dict": scheduler.state_dict()
                }
            )
            print(f"\n检查点已保存：{checkpoint_path}")
        
        # 更新进度条
        if not (val_dataloader and global_step % args.eval_freq == 0):
            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})
    
    # 计算平均 loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, global_step


def main():
    """主训练函数"""
    # ========== 解析参数 ==========
    args = parse_args()
    
    # ========== 设置随机种子 ==========
    set_seed(args.seed)
    
    # ========== 获取设备 ==========
    device = get_device()
    
    # ========== 创建输出目录 ==========
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 加载数据 ==========
    print(f"\n加载训练数据：{args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    print(f"训练文本长度：{len(train_text):,} 字符")
    
    # 创建训练数据加载器
    train_loader = create_dataloader_v1(
        train_text,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    
    print(f"训练批次数量：{len(train_loader)}")
    
    # 加载验证数据（可选）
    val_loader = None
    if args.val_data is not None:
        print(f"\n加载验证数据：{args.val_data}")
        with open(args.val_data, 'r', encoding='utf-8') as f:
            val_text = f.read()
        
        val_loader = create_dataloader_v1(
            val_text,
            tokenizer=args.tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            stride=args.stride,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )
        print(f"验证批次数量：{len(val_loader)}")
    
    # ========== 计算训练步数 ==========
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * args.epochs
    
    print(f"\n总训练步数：{num_training_steps:,}")
    print(f"每 epoch 步数：{steps_per_epoch:,}")
    
    # ========== 创建模型前清理显存 ==========
    clear_gpu_memory()
    if device.type == "cuda":
        print(f"可用显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"已分配显存：{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"缓存显存：{torch.cuda.memory_reserved() / 1024**3:.2f} GB\n")

    # ========== 创建模型 ==========
    config = get_model_config(args.model_config)
    model = create_model(
        config,
        hf_model_name=args.hf_model,
        ms_model_name=args.ms_model,
        model_source=args.model_source,
        device=device
    )

    # ========== 创建优化器和调度器 ==========
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        args.lr,
        args.min_lr_ratio,
        args.warmup_ratio,
        num_training_steps,
        args.weight_decay
    )
    
    # ========== 创建损失函数 ==========
    criterion = nn.CrossEntropyLoss()
    
    # ========== 初始化日志和检查点管理 ==========
    logger = TrainingLogger(
        log_dir=str(log_dir / datetime.now().strftime("%Y%m%d_%H%M%S")),
        model_config=config
    )
    
    checkpoint_manager = CheckpointManager(
        save_dir=str(output_dir),
        max_to_keep=3
    )
    
    # ========== 恢复训练（可选） ==========
    start_epoch = 0
    global_step = 0
    
    if args.resume is not None:
        print(f"\n从检查点恢复：{args.resume}")
        checkpoint = checkpoint_manager.load(model, optimizer, args.resume)
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)
        
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"恢复到 epoch {start_epoch}, step {global_step}")
    
    # ========== 开始训练 ==========
    print("\n" + "="*50)
    print("开始训练")
    print("="*50)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # 训练一个 epoch
        avg_loss, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            global_step=global_step,
            args=args,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            val_dataloader=val_loader,
            use_amp=args.mixed_precision
        )
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{args.epochs} 完成")
        print(f"  平均 Loss: {avg_loss:.4f}")
        print(f"  用时：{format_time(epoch_time)}")
        
        # 每个 epoch 结束后保存检查点
        if val_loader is not None:
            val_loss, perplexity = evaluate_model(model, val_loader, device)
            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                train_loss=avg_loss,
                val_loss=val_loss,
                is_best=True,
                extra_data={
                    "args": vars(args),
                    "scheduler_state_dict": scheduler.state_dict()
                }
            )
    
    # ========== 训练结束 ==========
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("训练完成!")
    print(f"总用时：{format_time(total_time)}")
    print(f"最终检查点：{output_dir / 'latest.pt'}")
    print("="*50)
    
    # 保存最终模型
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(args),
    }, final_model_path)
    print(f"最终模型已保存：{final_model_path}")
    
    # 关闭日志
    logger.close()


# 导入 datetime（用于日志目录命名）
from datetime import datetime


if __name__ == "__main__":
    main()
