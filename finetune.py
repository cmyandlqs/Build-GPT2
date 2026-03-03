"""
GPT-2 微调脚本

功能：
- 监督微调（SFT）- 文本分类等任务
- 指令微调（Instruction Tuning）- 遵循指令能力
- 支持从检查点或 HuggingFace 加载模型
- 全量微调（后续可扩展 LoRA）

使用示例:
    # 指令微调
    python finetune.py --checkpoint checkpoints/pt_model.pt \\
                       --data data/instruction-data.json \\
                       --task instruction --epochs 5
    
    # 从 HuggingFace 加载后微调
    python finetune.py --hf_model gpt2 --data data/instruction-data.json \\
                       --task instruction --epochs 3
    
    # 文本分类微调
    python finetune.py --checkpoint checkpoints/pt_model.pt \\
                       --data data/classification.tsv \\
                       --task classification --epochs 3
"""

import argparse
import os
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import GPTConfig, GPT_CONFIG_124M, get_model_config
from model import GPTModel, count_parameters
from data import create_instruction_dataloader, create_dataloader_v1
from utils import (
    CheckpointManager,
    TrainingLogger,
    evaluate_model,
    get_cosine_schedule_with_warmup,
    get_device,
    set_seed,
    format_time,
    load_weights_from_hf,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPT-2 微调脚本")
    
    # ========== 模型加载 ==========
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="预训练检查点路径"
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
        help="微调数据文件路径"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="验证数据文件路径"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["instruction", "classification"],
        default="instruction",
        help="微调任务类型"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["tiktoken", "bpe"],
        default="tiktoken",
        help="分词器类型"
    )
    
    # ========== 训练参数 ==========
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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
        default=512,
        help="序列最大长度"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率（微调通常用更小的学习率）"
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
    
    # ========== 检查点和日志 ==========
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_finetuned",
        help="输出目录"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs_finetune",
        help="TensorBoard 日志目录"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=50,
        help="评估频率（每 N 步）"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=200,
        help="保存检查点频率（每 N 步）"
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


def load_pretrained_model(
    checkpoint_path: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    ms_model_name: Optional[str] = None,
    model_source: str = "huggingface",
    device: torch.device = torch.device("cpu")
) -> Tuple[GPTModel, GPTConfig]:
    """
    加载预训练模型

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
        if hf_model_name == "gpt2":
            config = GPT_CONFIG_124M
        elif "gpt2-medium" in hf_model_name:
            from config import GPT_CONFIG_355M
            config = GPT_CONFIG_355M
        elif "gpt2-large" in hf_model_name:
            from config import GPT_CONFIG_774M
            config = GPT_CONFIG_774M
        elif "gpt2-xl" in hf_model_name:
            from config import GPT_CONFIG_1558M
            config = GPT_CONFIG_1558M
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

    # 打印参数量
    total_params = count_parameters(model)
    print(f"模型参数量：{total_params:,} ({total_params/1e6:.2f}M)")

    return model, config


def create_dataloaders(
    args: argparse.Namespace
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器
    
    Args:
        args: 命令行参数
        
    Returns:
        (train_loader, val_loader) 元组
    """
    print(f"\n加载训练数据：{args.data}")
    
    if args.task == "instruction":
        # 指令微调数据
        train_loader = create_instruction_dataloader(
            data_path=args.data,
            tokenizer=args.tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers
        )
        
        val_loader = None
        if args.val_data is not None:
            val_loader = create_instruction_dataloader(
                data_path=args.val_data,
                tokenizer=args.tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )
    
    elif args.task == "classification":
        # 预训练数据格式（简单文本）
        with open(args.data, 'r', encoding='utf-8') as f:
            train_text = f.read()
        
        train_loader = create_dataloader_v1(
            train_text,
            tokenizer=args.tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            stride=args.max_length // 2,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers
        )
        
        val_loader = None
        if args.val_data is not None:
            with open(args.val_data, 'r', encoding='utf-8') as f:
                val_text = f.read()
            
            val_loader = create_dataloader_v1(
                val_text,
                tokenizer=args.tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                stride=args.max_length // 2,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )
    else:
        raise ValueError(f"不支持的任务类型：{args.task}")
    
    print(f"训练批次数量：{len(train_loader)}")
    if val_loader is not None:
        print(f"验证批次数量：{len(val_loader)}")
    
    return train_loader, val_loader


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
        val_dataloader: 验证数据加载器
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
            attention_mask = None
        else:
            # 指令微调数据格式
            x = batch["input_ids"]
            y = batch["labels"]
            attention_mask = batch.get("attention_mask")
        
        x = x.to(device)
        y = y.to(device)
        
        # 自动混合精度
        if use_amp:
            with torch.amp.autocast(device.type):
                logits = model(x)
                
                # 计算 loss
                if y.dtype == torch.long:
                    loss = criterion(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )
                else:
                    # 处理 float 类型的标签
                    loss = criterion(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1).long()
                    )
                
                loss = loss / args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else loss
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % getattr(args, 'gradient_accumulation_steps', 1) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            # 标准训练
            logits = model(x)
            
            # 计算 loss
            if isinstance(batch, dict) and "labels" in batch:
                # 指令微调：labels 中 -100 的位置被忽略
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
            else:
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
            
            # 反向传播
            loss.backward()
            
            # 更新权重
            if (batch_idx + 1) % getattr(args, 'gradient_accumulation_steps', 1) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # 记录 loss
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        # 获取当前学习率
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录日志
        if global_step % 10 == 0:
            logger.log_train_step(
                step=global_step,
                loss=loss.item(),
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
                    "scheduler_state_dict": scheduler.state_dict(),
                    "task": args.task
                }
            )
            print(f"\n检查点已保存：{checkpoint_path}")
        
        # 更新进度条
        if not (val_dataloader and global_step % args.eval_freq == 0):
            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


def main():
    """主函数"""
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
    
    # ========== 加载预训练模型 ==========
    model, config = load_pretrained_model(
        checkpoint_path=args.checkpoint,
        hf_model_name=args.hf_model,
        ms_model_name=args.ms_model,
        model_source=args.model_source,
        device=device
    )
    
    # ========== 创建数据加载器 ==========
    train_loader, val_loader = create_dataloaders(args)
    
    # ========== 计算训练步数 ==========
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * args.epochs
    
    print(f"\n总训练步数：{num_training_steps:,}")
    print(f"每 epoch 步数：{steps_per_epoch:,}")
    
    # ========== 创建优化器和调度器 ==========
    # 微调通常使用较小的学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    
    # ========== 创建损失函数 ==========
    criterion = nn.CrossEntropyLoss()
    
    # ========== 初始化日志和检查点管理 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(
        log_dir=str(log_dir / f"{args.task}_{timestamp}"),
        model_config=config
    )
    
    checkpoint_manager = CheckpointManager(
        save_dir=str(output_dir / f"{args.task}_{timestamp}"),
        max_to_keep=3
    )
    
    # ========== 开始训练 ==========
    print("\n" + "="*50)
    print("开始微调")
    print(f"任务类型：{args.task}")
    print("="*50)
    
    start_time = time.time()
    global_step = 0
    
    for epoch in range(args.epochs):
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
                    "scheduler_state_dict": scheduler.state_dict(),
                    "task": args.task
                }
            )
    
    # ========== 训练结束 ==========
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("微调完成!")
    print(f"总用时：{format_time(total_time)}")
    print(f"最终检查点：{checkpoint_manager.save_dir / 'latest.pt'}")
    print("="*50)
    
    # 保存最终模型
    final_model_path = output_dir / f"finetuned_{args.task}_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "task": args.task,
    }, final_model_path)
    print(f"最终模型已保存：{final_model_path}")
    
    # 关闭日志
    logger.close()


if __name__ == "__main__":
    main()
