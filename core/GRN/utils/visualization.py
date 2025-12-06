"""可视化工具"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os


def visualize_results(
    input_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    可视化输入、预测和目标结果
    
    Args:
        input_tensor: 输入 [5, H, W] (Depth=1, Normal=3, Mask=1)
        pred_tensor: 预测 [4, H, W] (Depth=1, Normal=3)
        target_tensor: 目标 [4, H, W] (Depth=1, Normal=3)
        mask_tensor: 掩码 [H, W]
        save_path: 保存路径
        show: 是否显示
    """
    # 转换为numpy
    input_np = input_tensor.detach().cpu().numpy()
    pred_np = pred_tensor.detach().cpu().numpy()
    target_np = target_tensor.detach().cpu().numpy()
    mask_np = mask_tensor.detach().cpu().numpy()
    
    # 提取通道
    input_depth = input_np[0]
    input_normal = input_np[1:4].transpose(1, 2, 0)
    input_mask = input_np[4]
    
    pred_depth = pred_np[0]
    pred_normal = pred_np[1:4].transpose(1, 2, 0)
    
    target_depth = target_np[0]
    target_normal = target_np[1:4].transpose(1, 2, 0)
    
    # 创建图像
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # 输入
    axes[0, 0].imshow(input_depth, cmap='viridis')
    axes[0, 0].set_title('Input Depth')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow((input_normal + 1) / 2)
    axes[0, 1].set_title('Input Normal')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(input_mask, cmap='gray')
    axes[0, 2].set_title('Input Mask')
    axes[0, 2].axis('off')
    
    axes[0, 3].axis('off')
    
    # 预测
    axes[1, 0].imshow(pred_depth, cmap='viridis')
    axes[1, 0].set_title('Predicted Depth')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow((pred_normal + 1) / 2)
    axes[1, 1].set_title('Predicted Normal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(pred_depth - target_depth), cmap='hot')
    axes[1, 2].set_title('Depth Error')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(np.linalg.norm(pred_normal - target_normal, axis=2), cmap='hot')
    axes[1, 3].set_title('Normal Error')
    axes[1, 3].axis('off')
    
    # 目标
    axes[2, 0].imshow(target_depth, cmap='viridis')
    axes[2, 0].set_title('Target Depth')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow((target_normal + 1) / 2)
    axes[2, 1].set_title('Target Normal')
    axes[2, 1].axis('off')
    
    axes[2, 2].axis('off')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

