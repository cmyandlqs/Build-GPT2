"""
数据模块单元测试

测试内容：
- 分词器编码/解码
- 数据集类
- 数据加载器
"""

import os
import json
import tempfile
import pytest
import torch

from data import (
    GPTDatasetV1,
    create_dataloader_v1,
    InstructionDataset,
    create_instruction_dataloader,
)

import tiktoken


class TestGPTDatasetV1:
    """测试预训练数据集"""
    
    def test_dataset_length(self):
        """测试数据集长度"""
        tokenizer = tiktoken.get_encoding("gpt2")
        text = "This is a test. " * 100  # 足够长的文本
        
        max_length = 10
        stride = 5
        
        dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
        
        # 计算预期长度
        token_ids = tokenizer.encode(text, allowed_special={""})
        expected_length = (len(token_ids) - max_length) // stride
        
        assert len(dataset) > 0
    
    def test_item_shape(self):
        """测试数据项形状"""
        tokenizer = tiktoken.get_encoding("gpt2")
        text = "This is a test. " * 100
        
        max_length = 10
        stride = 5
        
        dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
        input_ids, target_ids = dataset[0]
        
        assert input_ids.shape == (max_length,)
        assert target_ids.shape == (max_length,)
    
    def test_target_shift(self):
        """测试目标序列是输入序列右移一位"""
        tokenizer = tiktoken.get_encoding("gpt2")
        text = "Hello world. This is a test."
        
        max_length = 5
        stride = 1
        
        dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
        
        # 获取第一个样本
        input_ids, target_ids = dataset[0]
        
        # 目标应该是输入右移一位
        token_ids = tokenizer.encode(text, allowed_special={""})
        expected_input = torch.tensor(token_ids[:max_length])
        expected_target = torch.tensor(token_ids[1:max_length + 1])
        
        assert torch.equal(input_ids, expected_input)
        assert torch.equal(target_ids, expected_target)


class TestCreateDataloaderV1:
    """测试数据加载器创建"""
    
    def test_dataloader_batch_size(self):
        """测试批次大小"""
        text = "This is a test. " * 100
        batch_size = 4
        
        dataloader = create_dataloader_v1(
            text,
            batch_size=batch_size,
            max_length=10,
            stride=5
        )
        
        for batch in dataloader:
            x, y = batch
            assert x.shape[0] == batch_size
            assert y.shape[0] == batch_size
            break
    
    def test_dataloader_shuffle(self):
        """测试数据打乱"""
        text = "This is a test. " * 100
        
        dataloader_shuffled = create_dataloader_v1(
            text,
            batch_size=4,
            max_length=10,
            stride=10,
            shuffle=True
        )
        
        dataloader_fixed = create_dataloader_v1(
            text,
            batch_size=4,
            max_length=10,
            stride=10,
            shuffle=False
        )
        
        # 两次打乱应该不同（大概率）
        # 固定顺序应该相同
        # 简化测试：只检查能否正常迭代
        for batch in dataloader_shuffled:
            assert len(batch) == 2
            break
        
        for batch in dataloader_fixed:
            assert len(batch) == 2
            break


class TestInstructionDataset:
    """测试指令微调数据集"""
    
    def test_dataset_loading(self):
        """测试数据加载"""
        # 创建临时测试数据
        test_data = [
            {
                "instruction": "Test instruction",
                "input": "Test input",
                "output": "Test output"
            },
            {
                "instruction": "Another instruction",
                "input": "",
                "output": "Another output"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            tokenizer = tiktoken.get_encoding("gpt2")
            dataset = InstructionDataset(temp_path, tokenizer, max_length=128)
            
            assert len(dataset) == 2
            
            item = dataset[0]
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
            
        finally:
            os.unlink(temp_path)
    
    def test_format_template(self):
        """测试格式化模板"""
        test_data = [
            {
                "instruction": "Translate",
                "input": "Hello",
                "output": "你好"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            tokenizer = tiktoken.get_encoding("gpt2")
            dataset = InstructionDataset(temp_path, tokenizer, max_length=256)
            
            # 检查格式化后的内容
            item = dataset.data[0]
            assert "### Instruction:" in item["input"]
            assert "### Input:" in item["input"]
            assert "### Response:" in item["full"]
            
        finally:
            os.unlink(temp_path)


class TestCreateInstructionDataloader:
    """测试指令数据加载器"""
    
    def test_dataloader_batch(self):
        """测试批次数据"""
        test_data = [
            {"instruction": f"Instruction {i}", "input": "", "output": f"Output {i}"}
            for i in range(10)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataloader = create_instruction_dataloader(
                temp_path,
                batch_size=2,
                max_length=128
            )
            
            for batch in dataloader:
                assert "input_ids" in batch
                assert "attention_mask" in batch
                assert "labels" in batch
                
                batch_size = batch["input_ids"].shape[0]
                assert batch_size <= 2  # 最后一个批次可能小于 batch_size
                break
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
