#!/usr/bin/env python3
"""
训练进度总结脚本
"""
import re
import os
from pathlib import Path

def parse_training_log(log_file="training.log"):
    """解析训练日志，提取关键信息"""
    if not os.path.exists(log_file):
        print(f"日志文件 {log_file} 不存在")
        return None
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取所有epoch的训练损失
    train_losses = re.findall(r'Epoch (\d+) - Train Loss \(Lgeo\): ([\d.]+)', content)
    val_losses = re.findall(r'Epoch (\d+) - Val Loss \(Lgeo\): ([\d.]+)', content)
    
    # 提取当前epoch和batch
    current_epoch_match = re.findall(r'Epoch (\d+) \[Train\]:\s+\d+%\|.*?\| (\d+)/(\d+)', content)
    
    result = {
        'train_losses': {int(ep): float(loss) for ep, loss in train_losses},
        'val_losses': {int(ep): float(loss) for ep, loss in val_losses},
        'current_epoch': None,
        'current_batch': None,
        'total_batches': None
    }
    
    if current_epoch_match:
        # 取最后一个匹配
        last_match = current_epoch_match[-1]
        result['current_epoch'] = int(last_match[0])
        result['current_batch'] = int(last_match[1])
        result['total_batches'] = int(last_match[2])
    
    return result

def main():
    print("=" * 60)
    print("GRN 训练进度总结")
    print("=" * 60)
    
    # 检查训练进程
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'python train.py' in result.stdout:
            print("✓ 训练进程正在运行")
        else:
            print("✗ 训练进程未运行")
    except:
        pass
    
    # 解析日志
    log_data = parse_training_log()
    if not log_data:
        return
    
    # 显示已完成epoch的损失
    if log_data['train_losses']:
        print("\n已完成 Epoch 的训练损失:")
        for epoch in sorted(log_data['train_losses'].keys()):
            train_loss = log_data['train_losses'][epoch]
            val_loss = log_data['val_losses'].get(epoch, 'N/A')
            print(f"  Epoch {epoch:2d}: Train={train_loss:.4f}, Val={val_loss}")
    
    # 显示当前进度
    if log_data['current_epoch'] is not None:
        epoch = log_data['current_epoch']
        batch = log_data['current_batch']
        total = log_data['total_batches']
        progress = (batch / total * 100) if total > 0 else 0
        print(f"\n当前进度:")
        print(f"  Epoch: {epoch}/100")
        print(f"  Batch: {batch}/{total} ({progress:.1f}%)")
        
        # 估算剩余时间
        if epoch < 100:
            remaining_epochs = 100 - epoch
            remaining_batches_in_epoch = total - batch
            total_remaining = remaining_epochs * total + remaining_batches_in_epoch
            # 假设每个batch约2.7秒
            estimated_seconds = total_remaining * 2.7
            estimated_hours = estimated_seconds / 3600
            print(f"  预计剩余时间: {estimated_hours:.1f} 小时")
    
    # 检查检查点
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            print(f"\n检查点文件 ({len(checkpoints)} 个):")
            for cp in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                size_mb = cp.stat().st_size / (1024 * 1024)
                print(f"  {cp.name} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

