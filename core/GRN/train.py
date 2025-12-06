"""
GRN自监督训练脚本
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional
from omegaconf import OmegaConf

from models import create_grn_model
from data import GRNDataset
from losses import GRNLoss
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.visualization import visualize_results


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    loss_dict_sum = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        input_tensor = batch['input'].to(device)
        target_tensor = batch['target'].to(device)
        mask_tensor = batch['mask'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(input_tensor)
        
        # 计算损失
        losses = criterion(output, target_tensor, mask_tensor)
        loss = losses['total_loss']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        for key, value in losses.items():
            if key not in loss_dict_sum:
                loss_dict_sum[key] = 0.0
            # 处理value可能是tensor或float的情况
            if isinstance(value, torch.Tensor):
                loss_dict_sum[key] += value.item()
            else:
                loss_dict_sum[key] += float(value)
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
        
        # 记录到tensorboard
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            for key, value in losses.items():
                # 处理value可能是tensor或float的情况
                if isinstance(value, torch.Tensor):
                    writer.add_scalar(f'Train/Batch{key}', value.item(), global_step)
                else:
                    writer.add_scalar(f'Train/Batch{key}', float(value), global_step)
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {key: value / len(dataloader) for key, value in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    visualize: bool = False,
    save_dir: Optional[str] = None
):
    """验证"""
    model.eval()
    total_loss = 0.0
    loss_dict_sum = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    for batch_idx, batch in enumerate(pbar):
        input_tensor = batch['input'].to(device)
        target_tensor = batch['target'].to(device)
        mask_tensor = batch['mask'].to(device)
        
        # 前向传播
        output = model(input_tensor)
        
        # 计算损失
        losses = criterion(output, target_tensor, mask_tensor)
        loss = losses['total_loss']
        
        # 累计损失
        total_loss += loss.item()
        for key, value in losses.items():
            if key not in loss_dict_sum:
                loss_dict_sum[key] = 0.0
            # 处理value可能是tensor或float的情况
            if isinstance(value, torch.Tensor):
                loss_dict_sum[key] += value.item()
            else:
                loss_dict_sum[key] += float(value)
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
        
        # 可视化第一个样本
        if visualize and batch_idx == 0 and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            visualize_results(
                input_tensor[0],
                output[0],
                target_tensor[0],
                mask_tensor[0],
                save_path=os.path.join(save_dir, f'epoch_{epoch}_val.png'),
                show=False
            )
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {key: value / len(dataloader) for key, value in loss_dict_sum.items()}
    
    # 记录到tensorboard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        for key, value in avg_loss_dict.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
    
    return avg_loss, avg_loss_dict


def main(config_path: str = 'configs/default.yaml'):
    """主训练函数"""
    # 加载配置
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        print(f"配置文件不存在: {config_path}，使用默认配置")
        config = OmegaConf.create({
            'data': {'root': './data', 'train_split': 'train', 'val_split': 'val', 
                    'image_size': 256, 'num_views': 8, 'degradation': {}},
            'model': {'in_channels': 5, 'out_channels': 4},
            'training': {'batch_size': 8, 'num_epochs': 100, 'learning_rate': 0.001,
                        'weight_decay': 0.0001, 'gradient_clip': 1.0},
            'loss': {'depth_l1_weight': 1.0, 'depth_l2_weight': 1.0, 
                    'normal_weight': 2.0, 'smoothness_weight': 0.1},
            'output': {'checkpoint_dir': './checkpoints', 'log_dir': './logs',
                      'save_interval': 10, 'val_interval': 5, 'visualize_interval': 20},
            'device': 'cuda', 'num_workers': 4, 'pin_memory': True
        })
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_grn_model(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels
    )
    model = model.to(device)
    print(f"模型参数量: {model.count_parameters() / 1e6:.2f}M")
    
    # 创建数据集
    train_dataset = GRNDataset(
        data_root=config.data.root,
        split=config.data.train_split,
        image_size=config.data.image_size,
        num_views=config.data.num_views,
        degradation_config=config.data.degradation
    )
    
    val_dataset = GRNDataset(
        data_root=config.data.root,
        split=config.data.val_split,
        image_size=config.data.image_size,
        num_views=config.data.num_views,
        degradation_config=config.data.degradation
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # 创建损失函数
    # 损失函数公式：Lgeo = Lrec + λsm Lsmooth + λnc Lnormal + λlap Llap
    criterion = GRNLoss(
        rec_weight=config.loss.rec_weight,
        smooth_weight=config.loss.smooth_weight,
        normal_weight=config.loss.normal_weight,
        laplacian_weight=config.loss.laplacian_weight
    )
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=0.00001
    )
    
    # 创建tensorboard writer
    os.makedirs(config.output.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.output.log_dir)
    
    # 训练循环
    best_val_loss = float('inf')
    start_epoch = 0
    
    for epoch in range(start_epoch, config.training.num_epochs):
        # 训练
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        print(f"\nEpoch {epoch} - Train Loss (Lgeo): {train_loss:.4f}")
        print(f"  Lrec: {train_loss_dict.get('Lrec', 0):.4f}")
        print(f"  Lsmooth: {train_loss_dict.get('Lsmooth', 0):.4f}")
        print(f"  Lnormal: {train_loss_dict.get('Lnormal', 0):.4f}")
        print(f"  Llap: {train_loss_dict.get('Llap', 0):.4f}")
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # 验证
        if (epoch + 1) % config.output.val_interval == 0:
            visualize = (epoch + 1) % config.output.visualize_interval == 0
            val_loss, val_loss_dict = validate(
                model, val_loader, criterion, device, epoch, writer,
                visualize=visualize,
                save_dir=os.path.join(config.output.log_dir, 'visualizations')
            )
            
            print(f"Epoch {epoch} - Val Loss (Lgeo): {val_loss:.4f}")
            print(f"  Lrec: {val_loss_dict.get('Lrec', 0):.4f}")
            print(f"  Lsmooth: {val_loss_dict.get('Lsmooth', 0):.4f}")
            print(f"  Lnormal: {val_loss_dict.get('Lnormal', 0):.4f}")
            print(f"  Llap: {val_loss_dict.get('Llap', 0):.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(config.output.checkpoint_dir, 'best_model.pth'),
                    additional_info={'val_loss': val_loss, 'train_loss': train_loss}
                )
        
        # 定期保存检查点
        if (epoch + 1) % config.output.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(config.output.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'),
                additional_info={'val_loss': val_loss if 'val_loss' in locals() else None}
            )
    
    writer.close()
    print("训练完成！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GRN训练脚本')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    args = parser.parse_args()
    
    main(args.config)

