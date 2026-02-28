# Build GPT2 项目需求文档

## 项目概述

从零开始构建一个**可训练、可微调、可评估**的 GPT-2 模型实现，参考 `LLMs-from-scratch` 项目的核心思路，代码逻辑清晰、易于学习，支持从 0 训练或加载 HuggingFace 预训练权重。

---

## 项目结构

```
Build_GPT2/
├── README.md            # 项目说明文档
├── requirements.txt     # 依赖列表（固定版本）
├── config.py            # 模型配置定义
├── model.py             # 模型架构定义
├── data.py              # 数据处理（分词器、数据集、数据加载器）
├── train.py             # 预训练脚本
├── finetune.py          # 微调脚本（SFT + 指令微调）
├── generate.py          # 文本生成脚本
├── evaluate.py          # 模型评估脚本
├── utils.py             # 工具函数（日志、Checkpoint 等）
├── tests/               # 测试代码
│   ├── test_model.py    # 模型单元测试
│   ├── test_data.py     # 数据模块单元测试
│   └── test_integration.py  # 集成测试
├── data/                # 示例数据
│   ├── the-verdict.txt      # 预训练示例文本
│   └── instruction-data.json # 指令微调示例数据
├── configs/             # 配置文件（可选扩展）
└── checkpoints/         # 模型保存目录（.gitignore）
```

---

## 功能需求

### 1. 模型配置 (`config.py`)

- 预设 **124M** 参数模型配置（GPT-2 Small）
- 支持通过参数自定义关键配置
- 配置项包括：
  ```python
  GPT_CONFIG_124M = {
      "vocab_size": 50257,
      "context_length": 1024,
      "emb_dim": 768,
      "n_heads": 12,
      "n_layers": 12,
      "drop_rate": 0.1,
      "qkv_bias": False,
  }
  ```

### 2. 模型架构 (`model.py`)

**核心模块：**
- `LayerNorm` - 层归一化
- `GELU` - GELU 激活函数
- `FeedForward` - 前馈神经网络
- `MultiHeadAttention` - 多头注意力机制（含因果掩码）
- `TransformerBlock` - Transformer 块
- `GPTModel` - 完整 GPT 模型

**位置编码：**
- 实现绝对位置编码（Absolute Position Embedding）
- 预留位置编码接口，便于后续替换为 RoPE

**文本生成：**
- `generate_text_simple()` - 贪婪解码
- 支持 Top-k Sampling
- 支持 Temperature 调节

### 3. 数据处理 (`data.py`)

**分词器：**
- 方案 A：使用 `tiktoken` GPT-2 分词器
- 方案 B：自实现 BPE 分词器（参考 `LLM-from-0/Chapter2-Data/origin_bpe.py`）
- 支持切换

**数据集类：**
- `GPTDatasetV1` - 预训练数据集（滑动窗口切分）
- `InstructionDataset` - 指令微调数据集（支持 JSON 格式）

**数据加载器：**
- `create_dataloader_v1()` - 预训练数据加载器
- `create_instruction_dataloader()` - 指令微调数据加载器

### 4. 预训练 (`train.py`)

**训练功能：**
- 单 GPU 训练
- 自动混合精度（AMP）
- 梯度裁剪（Gradient Clipping）
- 学习率调度（Cosine Decay + Warmup）
- AdamW 优化器 + Weight Decay

**日志与监控：**
- TensorBoard 日志记录
- 命令行实时输出（tqdm 进度条）
- 每 N 步打印训练 Loss

**Checkpoint：**
- 定期保存模型权重
- 保存训练状态（optimizer、epoch、step）
- 支持从 Checkpoint 恢复训练

**命令行参数示例：**
```bash
python train.py --data data/the-verdict.txt --model_config 124M \
                --epochs 10 --batch_size 4 --lr 5e-4 \
                --eval_freq 100 --checkpoint_freq 1000 \
                --output_dir checkpoints/
```

### 5. 微调 (`finetune.py`)

**支持模式：**
- 监督微调（SFT）- 文本分类等任务
- 指令微调（Instruction Tuning）- 遵循指令能力

**数据格式：**
- 参考 `LLMs-from-scratch` 的 `instruction-data.json` 格式
- JSON 列表，每个样本包含 `instruction`、`input`、`output`

**微调方式：**
- 全量微调（更新所有参数）
- 预留 LoRA 接口（后续扩展）

**命令行参数示例：**
```bash
# 监督微调
python finetune.py --checkpoint checkpoints/pt_model.pt \
                   --data data/instruction-data.json \
                   --task instruction --epochs 5 --batch_size 4

# 从 HuggingFace 加载权重后微调
python finetune.py --hf_model gpt2 --data data/instruction-data.json \
                   --task instruction --epochs 5
```

### 6. 文本生成 (`generate.py`)

**解码策略：**
- Greedy Decoding（贪婪解码）
- Top-k Sampling
- Temperature 调节

**功能：**
- 从 Checkpoint 加载模型
- 从 HuggingFace 加载模型
- 交互式生成 / 批量生成

**命令行参数示例：**
```bash
# 贪婪解码
python generate.py --checkpoint checkpoints/model.pt \
                   --prompt "Hello, I am" --max_new_tokens 50

# Top-k + Temperature
python generate.py --checkpoint checkpoints/model.pt \
                   --prompt "Once upon a time" \
                   --max_new_tokens 100 --top_k 50 --temperature 0.8
```

### 7. 模型评估 (`evaluate.py`)

**评估指标：**
- 训练/验证 Loss
- Perplexity（困惑度）

**定性评估：**
- 打印生成样本（人工评估文本质量）

**命令行参数示例：**
```bash
python evaluate.py --checkpoint checkpoints/model.pt \
                   --data data/val.txt \
                   --metrics loss perplexity generate \
                   --num_samples 5
```

### 8. 权重加载 (`utils.py`)

**支持来源：**
- 本地 Checkpoint 加载
- HuggingFace 模型加载（`transformers` 库）
  - 支持官方 GPT-2 模型（`gpt2`, `gpt2-medium`）
  - 支持 safetensors 格式

**工具函数：**
- `save_checkpoint()` - 保存检查点
- `load_checkpoint()` - 加载检查点
- `load_weights_from_hf()` - 从 HuggingFace 加载权重
- `count_parameters()` - 统计模型参数量

---

## 非功能需求

### 代码质量
- 关键逻辑有**中文注释**
- 每个函数/类有 **docstring**
- 复杂公式有**数学推导注释**（如 Attention 的缩放点积公式）
- 模块划分合理，单一职责

### 依赖管理
- 生成 `requirements.txt` 固定版本
- Python 版本：3.10+
- 核心依赖：
  ```
  torch>=2.2.2
  tiktoken>=0.5.1
  transformers>=4.30.0
  safetensors>=0.4.0
  tensorboard>=2.14.0
  tqdm>=4.66.0
  ```

### 测试
- **单元测试**：测试各模块功能正确性
  - 模型前向传播形状验证
  - 分词器编码/解码正确性
  - 数据集切片正确性
- **集成测试**：测试完整流程
  - 完整训练循环（小数据、少步数）
  - 生成流程验证

### 文档
- `README.md` 包含：
  - 项目简介
  - 快速开始（安装、运行示例）
  - 项目结构说明
  - 命令行参数说明
  - 参考资源链接

---

## 开发计划

### 阶段 1：基础功能
- [ ] `config.py` - 模型配置
- [ ] `model.py` - 模型架构
- [ ] `data.py` - 数据处理
- [ ] `train.py` - 预训练（基础训练循环）
- [ ] `generate.py` - 文本生成
- [ ] `README.md` - 基础文档

### 阶段 2：进阶功能
- [ ] `finetune.py` - 微调脚本
- [ ] `evaluate.py` - 评估脚本
- [ ] `utils.py` - 权重加载、Checkpoint
- [ ] TensorBoard 集成
- [ ] 完整命令行参数

### 阶段 3：完善优化
- [ ] `tests/` - 单元测试 + 集成测试
- [ ] `requirements.txt` - 依赖固定
- [ ] 代码审查与注释完善
- [ ] 示例数据准备
- [ ] 最终文档

---

## 参考资源

- **主要参考**: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- **GPT-2 论文**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **HuggingFace GPT-2**: https://huggingface.co/docs/transformers/model_doc/gpt2

---

## 命令行接口总览

| 脚本 | 功能 | 主要参数 |
|------|------|----------|
| `train.py` | 预训练 | `--data`, `--epochs`, `--batch_size`, `--lr` |
| `finetune.py` | 微调 | `--checkpoint`/`--hf_model`, `--data`, `--task` |
| `generate.py` | 文本生成 | `--checkpoint`, `--prompt`, `--top_k`, `--temperature` |
| `evaluate.py` | 评估 | `--checkpoint`, `--data`, `--metrics` |

---

## 备注

- 版本控制：完成后考虑是否设为独立 git 仓库
- 项目命名：`build-gpt2`（文件夹保持 `Build_GPT2`）
- 代码风格：参考业界主流实践，命名规范易懂
