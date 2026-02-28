"""
集成测试

测试完整流程：
- 模型训练循环
- 文本生成流程
- 保存和加载
"""

import os
import tempfile
import pytest
import torch
import shutil

from config import GPTConfig, GPT_CONFIG_124M
from model import GPTModel
from data import create_dataloader_v1
from utils import CheckpointManager, set_seed


class TestTrainingLoop:
    """测试训练循环"""
    
    def test_single_step(self):
        """测试单步训练"""
        set_seed(42)
        
        # 创建小型模型
        config = GPTConfig(
            vocab_size=50257,
            context_length=64,
            emb_dim=128,
            n_heads=4,
            n_layers=2,
            drop_rate=0.1,
        )
        
        model = GPTModel(config)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 创建测试数据
        text = "This is a test. " * 100
        dataloader = create_dataloader_v1(
            text,
            batch_size=2,
            max_length=32,
            stride=16
        )
        
        # 单步训练
        for batch in dataloader:
            x, y = batch
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            # 检查 loss 是有限值
            assert torch.isfinite(loss)
            break
    
    def test_single_epoch(self):
        """测试单个 epoch"""
        set_seed(42)
        
        config = GPTConfig(
            vocab_size=50257,
            context_length=32,
            emb_dim=64,
            n_heads=2,
            n_layers=1,
            drop_rate=0.0,  # 禁用 dropout 以便测试
        )
        
        model = GPTModel(config)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        text = "This is a test. " * 50
        dataloader = create_dataloader_v1(
            text,
            batch_size=2,
            max_length=16,
            stride=8
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            x, y = batch
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # 检查平均 loss 是有限值
        assert torch.isfinite(torch.tensor(avg_loss))


class TestGeneration:
    """测试文本生成"""
    
    def test_generation_basic(self):
        """测试基础生成功能"""
        set_seed(42)
        
        config = GPTConfig(
            vocab_size=50257,
            context_length=64,
            emb_dim=128,
            n_heads=4,
            n_layers=2,
        )
        
        model = GPTModel(config)
        model.eval()
        
        # 测试输入
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=1.0,
                top_k=None,
                greedy=True
            )
        
        # 检查输出长度
        assert generated.shape[1] == 10 + 20
    
    def test_generation_temperature(self):
        """测试温度参数"""
        set_seed(42)
        
        config = GPTConfig(
            vocab_size=50257,
            context_length=64,
            emb_dim=128,
            n_heads=4,
            n_layers=2,
        )
        
        model = GPTModel(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        # 不同温度应该产生不同结果（大概率）
        with torch.no_grad():
            gen_low = model.generate(input_ids, max_new_tokens=10, temperature=0.5, greedy=False)
            gen_high = model.generate(input_ids, max_new_tokens=10, temperature=1.5, greedy=False)
        
        # 形状应该相同
        assert gen_low.shape == gen_high.shape


class TestCheckpoint:
    """测试检查点保存和加载"""
    
    def test_save_and_load(self):
        """测试保存和加载"""
        set_seed(42)
        
        config = GPTConfig(
            vocab_size=50257,
            context_length=32,
            emb_dim=64,
            n_heads=2,
            n_layers=1,
        )
        
        model = GPTModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            checkpoint_manager = CheckpointManager(temp_dir, max_to_keep=3)
            
            # 保存检查点
            path = checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=0,
                step=100,
                train_loss=2.5
            )
            
            assert os.path.exists(path)
            
            # 加载检查点
            new_model = GPTModel(config)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            
            checkpoint = checkpoint_manager.load(new_model, new_optimizer, path)
            
            assert checkpoint["step"] == 100
            assert checkpoint["train_loss"] == 2.5
            
            # 验证模型权重相同
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.equal(p1, p2)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_latest_checkpoint(self):
        """测试最新检查点"""
        set_seed(42)
        
        config = GPTConfig(
            vocab_size=50257,
            context_length=32,
            emb_dim=64,
            n_heads=2,
            n_layers=1,
        )
        
        model = GPTModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            checkpoint_manager = CheckpointManager(temp_dir, max_to_keep=3)
            
            # 保存多个检查点
            checkpoint_manager.save(model, optimizer, 0, 100, 2.5)
            checkpoint_manager.save(model, optimizer, 0, 200, 2.3)
            checkpoint_manager.save(model, optimizer, 1, 300, 2.1)
            
            # 检查 latest.pt 存在
            latest_path = os.path.join(temp_dir, "latest.pt")
            assert os.path.exists(latest_path)
            
            # 加载最新检查点
            checkpoint = checkpoint_manager.load(model, optimizer)
            assert checkpoint["step"] == 300
            
        finally:
            shutil.rmtree(temp_dir)


class TestIntegration:
    """完整集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程：训练→保存→加载→生成"""
        set_seed(42)
        
        # 1. 创建小型模型
        config = GPTConfig(
            vocab_size=50257,
            context_length=32,
            emb_dim=64,
            n_heads=2,
            n_layers=1,
        )
        
        model = GPTModel(config)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 2. 训练几个步骤
        text = "This is a test. " * 50
        dataloader = create_dataloader_v1(
            text,
            batch_size=2,
            max_length=16,
            stride=8
        )
        
        num_steps = 0
        for batch in dataloader:
            x, y = batch
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            num_steps += 1
            if num_steps >= 5:
                break
        
        # 3. 保存模型
        temp_dir = tempfile.mkdtemp()
        
        try:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "step": num_steps,
            }, checkpoint_path)
            
            # 4. 加载模型
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            loaded_model = GPTModel(config)
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            loaded_model.eval()
            
            # 5. 生成测试
            input_ids = torch.randint(0, config.vocab_size, (1, 10))
            
            with torch.no_grad():
                generated = loaded_model.generate(
                    input_ids,
                    max_new_tokens=10,
                    greedy=True
                )
            
            assert generated.shape[1] == 20
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
