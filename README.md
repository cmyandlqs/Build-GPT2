# Build GPT2

从零开始构建 GPT-2 模型的完整实现，支持预训练、微调和文本生成。参考 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 项目的核心思路，代码逻辑清晰、易于学习。

## 特点

- 🎯 **完整流程**: 数据处理 → 模型构建 → 预训练 → 微调 → 评估 → 生成
- 📦 **模块化设计**: 清晰的模块划分，易于理解和扩展
- 📝 **中文注释**: 关键逻辑和复杂公式均有详细中文注释
- 🔄 **灵活配置**: 支持多种模型规模（124M/355M/774M/1558M）
- 💾 **权重加载**: 支持从 HuggingFace 加载预训练 GPT-2 权重
- 🧪 **完整测试**: 包含单元测试和集成测试

## 快速开始

### 安装依赖

```bash
cd Build_GPT2
pip install -r requirements.txt
```

### 预训练

```bash
# 基础训练（使用示例数据）
python train.py --data data/the-verdict.txt --epochs 10 --batch_size 4

# 查看完整参数
python train.py --help
```

### 文本生成

```bash
# 从检查点生成
python generate.py --checkpoint checkpoints/latest.pt --prompt "Hello, I am"

# 从 HuggingFace 加载模型生成
python generate.py --hf_model gpt2 --prompt "Once upon a time" --top_k 50 --temperature 0.8

# 从 ModelScope 加载模型生成
python generate.py --ms_model AI-ModelScope/gpt2 --prompt "Hello, I am"

# 交互式模式
python generate.py --checkpoint checkpoints/latest.pt --interactive
```

### 微调

```bash
# 指令微调
python finetune.py --checkpoint checkpoints/pt_model.pt \
                   --data data/instruction-data.json \
                   --task instruction --epochs 5

# 从 HuggingFace 加载后微调
python finetune.py --hf_model gpt2 --data data/instruction-data.json \
                   --task instruction --epochs 3

# 从 ModelScope 加载后微调
python finetune.py --ms_model AI-ModelScope/gpt2 --data data/instruction-data.json \
                   --task instruction --epochs 3
```

### 评估

```bash
# 计算 Loss 和 Perplexity
python evaluate.py --checkpoint checkpoints/model.pt --data data/val.txt

# 生成样本定性评估
python evaluate.py --checkpoint checkpoints/model.pt \
                   --data data/val.txt \
                   --metrics loss perplexity generate --num_samples 5
```

### 从 ModelScope 加载

本项目支持从 ModelScope（魔搭）加载 GPT-2 模型权重：

```bash
# 指定从 ModelScope 加载
python generate.py --hf_model gpt2 --model_source modelscope --prompt "Hello"

# 或直接使用 ms_model 参数
python generate.py --ms_model AI-ModelScope/gpt2 --prompt "Hello"

# 预训练时从 ModelScope 加载权重
python train.py --data data/the-verdict.txt --ms_model AI-ModelScope/gpt2 --epochs 5
```

支持的 ModelScope 模型：
- `AI-ModelScope/gpt2` (124M)
- `AI-ModelScope/gpt2-medium` (355M)
- `AI-ModelScope/gpt2-large` (774M)
- `AI-ModelScope/gpt2-xl` (1558M)

## 项目结构

```
Build_GPT2/
├── config.py              # 模型配置定义
├── model.py               # 模型架构（LayerNorm, Attention, Transformer, GPT）
├── data.py                # 数据处理（分词器、数据集、数据加载器）
├── utils.py               # 工具函数（Checkpoint、权重加载、日志）
├── train.py               # 预训练脚本
├── finetune.py            # 微调脚本（SFT + 指令微调）
├── generate.py            # 文本生成脚本
├── evaluate.py            # 模型评估脚本
├── requirements.txt       # 依赖列表
├── README.md              # 项目说明
├── data/                  # 示例数据
│   ├── the-verdict.txt        # 预训练示例文本
│   └── instruction-data.json  # 指令微调示例数据
├── tests/               # 测试代码
│   ├── test_model.py    # 模型单元测试
│   ├── test_data.py     # 数据模块测试
│   └── test_integration.py  # 集成测试
├── checkpoints/         # 模型保存目录（需手动创建）
└── logs/                # TensorBoard 日志目录（需手动创建）
```

## 模块说明

### `config.py` - 模型配置

定义 GPT-2 模型配置，支持多种规模：

```python
from config import GPT_CONFIG_124M, get_model_config

# 使用预定义配置
config = GPT_CONFIG_124M

# 或获取特定配置
config = get_model_config("355M")
```

| 配置 | 参数量 | emb_dim | n_heads | n_layers |
|------|--------|---------|---------|----------|
| 124M | 124M   | 768     | 12      | 12       |
| 355M | 355M   | 1024    | 16      | 24       |
| 774M | 774M   | 1280    | 20      | 36       |
| 1558M| 1558M  | 1600    | 25      | 48       |

### `model.py` - 模型架构

核心组件：
- `LayerNorm`: 层归一化
- `GELU`: GELU 激活函数
- `FeedForward`: 前馈神经网络
- `MultiHeadAttention`: 多头注意力机制
- `TransformerBlock`: Transformer 块
- `GPTModel`: 完整 GPT 模型

```python
from config import GPT_CONFIG_124M
from model import GPTModel

model = GPTModel(GPT_CONFIG_124M)
```

### `data.py` - 数据处理

支持两种分词器：
- **tiktoken**: GPT-2 官方分词器（推荐）
- **BPE**: 自实现 BPE 分词器（学习用）

数据集类：
- `GPTDatasetV1`: 预训练数据集（滑动窗口）
- `InstructionDataset`: 指令微调数据集

### `utils.py` - 工具函数

- `CheckpointManager`: 检查点管理（保存/加载）
- `TrainingLogger`: 训练日志（TensorBoard + 命令行）
- `load_weights_from_hf`: 从 HuggingFace 加载权重
- `evaluate_model`: 模型评估（Loss + Perplexity）
- `get_cosine_schedule_with_warmup`: 学习率调度器

## 命令行参数

### `train.py` 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 训练数据文件 | 必需 |
| `--model_config` | 模型配置 (124M/355M/...) | 124M |
| `--epochs` | 训练轮数 | 10 |
| `--batch_size` | 批次大小 | 4 |
| `--lr` | 学习率 | 5e-4 |
| `--hf_model` | HuggingFace 模型名 | - |
| `--ms_model` | ModelScope 模型名 | - |
| `--model_source` | 模型来源 (huggingface/modelscope) | huggingface |
| `--output_dir` | 输出目录 | checkpoints |
| `--resume` | 从检查点恢复 | None |
| `--mixed_precision` | 混合精度训练 | False |

### `generate.py` 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型检查点路径 | - |
| `--hf_model` | HuggingFace 模型名 | - |
| `--ms_model` | ModelScope 模型名 | - |
| `--model_source` | 模型来源 (huggingface/modelscope) | huggingface |
| `--prompt` | 生成提示 | "Hello, I am" |
| `--max_new_tokens` | 最大生成 token 数 | 100 |
| `--temperature` | 温度参数 | 1.0 |
| `--top_k` | Top-k 采样 | None |
| `--interactive` | 交互模式 | False |

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_model.py -v

# 运行集成测试
pytest tests/test_integration.py -v
```

## 示例输出

### 训练输出
```
==================================================
开始训练
==================================================
Epoch 1/10 | Step    100 | Loss: 2.5432 | LR: 5.00e-04
Epoch 1/10 | Step    200 | Loss: 2.3156 | LR: 4.95e-04
...
```

### 生成输出
```
==================================================
生成配置
==================================================
提示文本：Hello, I am
最大新 token 数：50
温度：0.8
Top-k: 50
==================================================

生成结果
==================================================
Hello, I am writing to you from the heart of the 
Riviera, where I have been staying for the past 
few weeks...
==================================================
```

## 硬件要求

- **最低**: CPU（训练较慢）
- **推荐**: NVIDIA GPU (8GB+ 显存)
- **最佳**: NVIDIA GPU (24GB+ 显存，支持更大模型)

## 参考资源

- **主要参考**: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka
- **GPT-2 论文**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **HuggingFace GPT-2**: https://huggingface.co/docs/transformers/model_doc/gpt2

## 许可证

本项目代码仅供学习和研究使用。

## 常见问题

### Q: 如何修改模型配置？
编辑 `config.py` 中的 `GPTConfig` 或创建自定义配置。

### Q: 如何使用自己的数据训练？
准备纯文本文件（.txt），通过 `--data` 参数指定路径。

### Q: 训练中断后如何恢复？
使用 `--resume checkpoints/latest.pt` 从最新检查点恢复。

### Q: 如何调整生成文本的随机性？
调整 `--temperature` 参数：
- `< 1`: 更确定（保守）
- `> 1`: 更随机（创造性）
