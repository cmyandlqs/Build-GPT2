"""
数据处理模块

提供 GPT-2 模型所需的数据处理功能：
- 分词器（tiktoken 和自实现 BPE）
- 预训练数据集
- 指令微调数据集
- 数据加载器创建
"""

import json
import os
import re
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Union

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from config import GPTConfig


#####################################
# BPE 分词器（自实现）
#####################################

@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """
    返回 UTF-8 字节到 Unicode 字符的映射
    
    BPE 算法在 Unicode 字符串上工作，为了避免未知 token (UNK)，
    需要建立字节到 Unicode 的查找表。这个映射确保所有可能的
    UTF-8 字节都能被处理。
    
    Returns:
        字节值到 Unicode 字符的字典
    """
    # 基本字符范围：ASCII 可打印字符和扩展 ASCII
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    
    # 为剩余的字节值分配 Unicode 码位
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str, ...]) -> set:
    """
    获取单词中相邻字符对的集合
    
    Args:
        word: 字符元组
        
    Returns:
        字符对的集合
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BPETokenizer:
    """
    自实现的 Byte Pair Encoding (BPE) 分词器
    
    BPE 是一种子词分词算法，通过迭代合并最频繁的字符对来构建词汇表。
    这个实现参考了 GPT-2 的分词方式，支持编码和解码。
    
    算法流程:
    1. 将输入文本转换为字节序列
    2. 字节映射到 Unicode 字符
    3. 使用正则表达式分割成初始 token
    4. 迭代合并最频繁的字符对
    5. 映射到词汇表 ID
    
    Attributes:
        encoder: token 到 ID 的映射
        decoder: ID 到 token 的映射
        bpe_ranks: BPE 合并规则的优先级
        cache: 缓存已处理的 token
        pat: 正则表达式模式，用于 token 分割
    """
    
    def __init__(self, encoder: Dict[str, int], bpe_merges: List[Tuple[str, str]]):
        """
        初始化 BPE 分词器
        
        Args:
            encoder: token 字符串到 ID 的字典
            bpe_merges: BPE 合并规则列表
        """
        self.encoder = encoder
        self.decoder = {v: k for k, v in encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        
        # 正则表达式模式，匹配不同类型的 token
        # 包括：缩写、单词、数字、标点、空格
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        # 字节到 Unicode 的映射
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
    def bpe(self, token: str) -> str:
        """
        对单个 token 应用 BPE 算法
        
        迭代合并字符对，直到无法合并或达到词汇表限制。
        使用缓存避免重复计算。
        
        Args:
            token: 输入 token 字符串
            
        Returns:
            BPE 处理后的字符串（用空格分隔的单元）
        """
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # 找到优先级最高的字符对（排名最小）
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            # 如果没有更多合并规则，退出
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    # 找到 first 字符的位置
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    # 找不到则添加剩余字符
                    new_word.extend(word[i:])
                    break
                
                # 如果找到字符对，则合并
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            
            # 如果只剩一个单元，停止
            if len(word) == 1:
                break
            
            # 重新获取字符对
            pairs = get_pairs(word)
        
        # 用空格连接字符单元
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 BPE token ID 序列
        
        Args:
            text: 输入文本
            
        Returns:
            token ID 列表
        """
        bpe_tokens = []
        
        # 使用正则表达式分割文本
        for token in re.findall(self.pat, text):
            # 转换为字节，再映射到 Unicode 字符
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            
            # 应用 BPE 并转换为 ID
            bpe_token = self.bpe(token)
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in bpe_token.split(' ')
            )
        
        return bpe_tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        将 token ID 序列解码为文本
        
        Args:
            tokens: token ID 列表
            
        Returns:
            解码后的文本
        """
        # ID → 编码字符
        text = ''.join(self.decoder[token] for token in tokens)
        
        # Unicode 字符 → 字节 → UTF-8 文本
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text


def load_bpe_tokenizer(models_dir: str, model_name: str = "gpt2") -> BPETokenizer:
    """
    从文件加载 BPE 分词器
    
    Args:
        models_dir: 模型文件目录
        model_name: 模型名称
        
    Returns:
        BPETokenizer 实例
    """
    encoder_path = os.path.join(models_dir, model_name, 'encoder.json')
    bpe_path = os.path.join(models_dir, model_name, 'vocab.bpe')
    
    with open(encoder_path, 'r', encoding='utf-8') as f:
        encoder = json.load(f)
    
    with open(bpe_path, 'r', encoding='utf-8') as f:
        bpe_data = f.read()
    
    # 解析 BPE 合并规则（跳过第一行注释和最后一行空行）
    bpe_merges = [
        tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
    ]
    
    return BPETokenizer(encoder=encoder, bpe_merges=bpe_merges)


#####################################
# 预训练数据集
#####################################

class GPTDatasetV1(Dataset):
    """
    GPT 预训练数据集
    
    使用滑动窗口将文本切分为重叠的序列块。
    每个样本包含输入序列和目标序列，目标序列是输入序列向右移动一位。
    
    例如，对于文本 "The quick brown fox"：
    - 输入：[The, quick, brown]
    - 目标：[quick, brown, fox]
    
    这种设计让模型学习预测下一个 token。
    
    Attributes:
        input_ids: 输入 token ID 列表
        target_ids: 目标 token ID 列表
    """
    
    def __init__(
        self,
        txt: str,
        tokenizer: Union[tiktoken.Encoding, BPETokenizer],
        max_length: int,
        stride: int
    ):
        """
        初始化数据集
        
        Args:
            txt: 原始文本数据
            tokenizer: 分词器（tiktoken 或 BPE）
            max_length: 每个序列的最大长度
            stride: 滑动窗口步长
        """
        self.input_ids = []
        self.target_ids = []
        
        # 分词
        if isinstance(tokenizer, BPETokenizer):
            token_ids = tokenizer.encode(txt)
        else:
            token_ids = tokenizer.encode(txt, allowed_special={""})
        
        # 滑动窗口切分
        for i in range(0, len(token_ids) - max_length, stride):
            # 输入块
            input_chunk = token_ids[i:i + max_length]
            # 目标块（向右移动一位）
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的样本
        
        Args:
            idx: 索引
            
        Returns:
            (input_ids, target_ids) 元组
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    tokenizer: Union[tiktoken.Encoding, BPETokenizer, str] = "tiktoken",
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    创建预训练数据加载器
    
    Args:
        txt: 原始文本数据
        tokenizer: 分词器，可以是：
            - "tiktoken": 使用 tiktoken GPT-2 分词器
            - "bpe": 使用自实现 BPE 分词器
            - 已初始化的分词器实例
        batch_size: 批次大小
        max_length: 序列最大长度
        stride: 滑动窗口步长
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后一个不完整的批次
        num_workers: 数据加载进程数
        
    Returns:
        PyTorch DataLoader
    """
    # 初始化分词器
    if tokenizer == "tiktoken":
        tokenizer = tiktoken.get_encoding("gpt2")
    elif tokenizer == "bpe":
        # 需要指定模型目录
        raise ValueError("使用 BPE 分词器需要提供 models_dir 参数")
    
    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader


#####################################
# 指令微调数据集
#####################################

class InstructionDataset(Dataset):
    """
    指令微调数据集
    
    用于监督微调（SFT）和指令微调的数据集。
    支持多种数据格式，包括 instruction-input-output 格式。
    
    数据格式示例:
    ```json
    [
        {
            "instruction": "翻译以下句子",
            "input": "Hello, how are you?",
            "output": "你好，你好吗？"
        }
    ]
    ```
    
    格式化模板:
        ### Instruction:
        {instruction}
        
        ### Input:
        {input}
        
        ### Response:
        {output}
    
    Attributes:
        data: 处理后的数据列表
        tokenizer: 分词器
        max_length: 最大序列长度
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Union[tiktoken.Encoding, BPETokenizer],
        max_length: int = 1024,
        format_template: Optional[str] = None
    ):
        """
        初始化指令数据集
        
        Args:
            data_path: JSON 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            format_template: 格式化模板，None 使用默认模板
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 默认格式化模板
        if format_template is None:
            self.format_template = (
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n{output}"
            )
        else:
            self.format_template = format_template
        
        # 加载数据
        self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> None:
        """
        加载 JSON 数据文件
        
        Args:
            data_path: 文件路径
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 处理每个样本
        for item in raw_data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            
            # 格式化输入文本
            formatted_input = self.format_template.format(
                instruction=instruction,
                input=input_text,
                output=""
            )
            
            # 格式化完整文本（包含输出）
            formatted_full = self.format_template.format(
                instruction=instruction,
                input=input_text,
                output=output_text
            )
            
            self.data.append({
                "input": formatted_input,
                "full": formatted_full
            })
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取指定索引的样本
        
        Args:
            idx: 索引
            
        Returns:
            包含 input_ids、attention_mask、labels 的字典
        """
        item = self.data[idx]
        
        # 编码完整文本
        if isinstance(self.tokenizer, BPETokenizer):
            encoded_full = self.tokenizer.encode(item["full"])
            encoded_input = self.tokenizer.encode(item["input"])
        else:
            encoded_full = self.tokenizer.encode(item["full"], allowed_special={""})
            encoded_input = self.tokenizer.encode(item["input"], allowed_special={""})
        
        # 截断到最大长度
        encoded_full = encoded_full[:self.max_length]
        
        # 创建 attention mask（1 表示有效 token，0 表示 padding）
        attention_mask = [1] * len(encoded_full)
        
        # 创建 labels（用于计算 loss）
        # input 部分的 label 设为 -100（忽略）
        # output 部分使用实际 token ID
        labels = [-100] * len(encoded_input) + encoded_full[len(encoded_input):]
        
        # Padding 到最大长度
        pad_length = self.max_length - len(encoded_full)
        if pad_length > 0:
            encoded_full = encoded_full + [0] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            labels = labels + [-100] * pad_length
        
        return {
            "input_ids": torch.tensor(encoded_full, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def create_instruction_dataloader(
    data_path: str,
    tokenizer: Union[tiktoken.Encoding, BPETokenizer, str] = "tiktoken",
    batch_size: int = 4,
    max_length: int = 1024,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    创建指令微调数据加载器
    
    Args:
        data_path: JSON 数据文件路径
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 序列最大长度
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后一个不完整的批次
        num_workers: 数据加载进程数
        
    Returns:
        PyTorch DataLoader
    """
    # 初始化分词器
    if tokenizer == "tiktoken":
        tokenizer = tiktoken.get_encoding("gpt2")
    elif tokenizer == "bpe":
        raise ValueError("使用 BPE 分词器需要提供 models_dir 参数")
    
    # 创建数据集
    dataset = InstructionDataset(data_path, tokenizer, max_length)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn  # 自定义 collate 函数
    )
    
    return dataloader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数，用于批处理指令数据
    
    Args:
        batch: 样本列表
        
    Returns:
        批处理后的字典
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


#####################################
# 文本分类数据集（用于监督微调）
#####################################

class ClassificationDataset(Dataset):
    """
    文本分类数据集
    
    用于文本分类任务的监督微调。
    支持二分类和多分类任务。
    
    数据格式:
        每行一个样本：文本\t标签
    
    Attributes:
        texts: 文本列表
        labels: 标签列表
        tokenizer: 分词器
        max_length: 最大序列长度
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Union[tiktoken.Encoding, BPETokenizer],
        max_length: int = 512,
        label2id: Optional[Dict[str, int]] = None
    ):
        """
        初始化分类数据集
        
        Args:
            data_path: 数据文件路径（TSV 格式）
            tokenizer: 分词器
            max_length: 最大序列长度
            label2id: 标签到 ID 的映射，None 则自动构建
        """
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {}
        self.id2label = {}
        
        self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> None:
        """加载 TSV 数据文件"""
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过表头（如果有）
        start_idx = 0
        if lines[0].strip().lower().startswith('text'):
            start_idx = 1
        
        for line in lines[start_idx:]:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                text, label = parts[0], parts[1]
                self.texts.append(text)
                
                # 构建标签映射
                if label not in self.label2id:
                    self.label2id[label] = len(self.label2id)
                
                self.labels.append(self.label2id[label])
        
        # 构建反向映射
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        if isinstance(self.tokenizer, BPETokenizer):
            encoded = self.tokenizer.encode(text)
        else:
            encoded = self.tokenizer.encode(text, allowed_special={""})
        
        # 截断
        encoded = encoded[:self.max_length - 1]  # 留一个位置给 [CLS] 或特殊 token
        
        # 添加序列开始 token（如果有）
        # GPT-2 没有 [CLS]，直接使用编码后的 token
        
        # Padding
        pad_length = self.max_length - len(encoded)
        if pad_length > 0:
            encoded = encoded + [0] * pad_length
        
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }


if __name__ == "__main__":
    # ========== 测试分词器 ==========
    print("=" * 50)
    print("测试 tiktoken 分词器")
    print("=" * 50)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, I am a GPT-2 model."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"原文：{text}")
    print(f"编码：{encoded}")
    print(f"解码：{decoded}")
    
    # ========== 测试数据集 ==========
    print("\n" + "=" * 50)
    print("测试预训练数据集")
    print("=" * 50)
    
    sample_text = "This is a sample text for testing the dataset. " * 100
    # print(f'样本文本：{sample_text}')
    dataloader = create_dataloader_v1(
        sample_text,
        batch_size=2,
        max_length=10,
        stride=5
    )
    
    for batch in dataloader:
        x, y = batch
        print(f"输入形状：{x.shape}, 目标形状：{y.shape}")
        print(f"输入：{x[0].tolist()}")
        print(f"目标：{y[0].tolist()}")
        break
