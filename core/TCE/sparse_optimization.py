"""
显式稀疏优化模块

对生成的纹理贴图进行 Patch 级别的显式优化，而非仅依赖隐式扩散先验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SparsePatchOptimizer(nn.Module):
    """
    显式稀疏优化器
    
    对纹理贴图进行 Patch 级别的显式优化
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        device: str = "cuda"
    ):
        """
        初始化稀疏优化器
        
        Args:
            patch_size: Patch 大小
            device: 计算设备
        """
        super().__init__()
        self.patch_size = patch_size
        self.device = device
    
    def extract_patches(
        self, 
        texture: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        从纹理中提取 Patch
        
        Args:
            texture: 纹理贴图 [C, H, W]
            
        Returns:
            patches: Patch 张量 [num_patches, C, patch_size, patch_size]
            grid_shape: Patch 网格形状 (num_patches_h, num_patches_w)
        """
        C, H, W = texture.shape
        
        # 计算 Patch 网格
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # 调整纹理尺寸以匹配 Patch 网格
        H_adj = num_patches_h * self.patch_size
        W_adj = num_patches_w * self.patch_size
        texture_adj = F.interpolate(
            texture.unsqueeze(0),
            size=(H_adj, W_adj),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # 提取 Patch
        patches = texture_adj.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )  # [C, num_patches_h, num_patches_w, patch_size, patch_size]
        
        patches = patches.contiguous().view(
            C, num_patches_h * num_patches_w, self.patch_size, self.patch_size
        )
        patches = patches.permute(1, 0, 2, 3)  # [num_patches, C, patch_size, patch_size]
        
        return patches, (num_patches_h, num_patches_w)
    
    def compute_patch_consistency(
        self, 
        patches: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Patch 之间的一致性
        
        Args:
            patches: Patch 张量 [num_patches, C, patch_size, patch_size]
            
        Returns:
            一致性损失（标量）
        """
        num_patches = patches.shape[0]
        
        if num_patches < 2:
            return torch.tensor(0.0, device=self.device)
        
        # 计算相邻 Patch 之间的边界一致性
        # 这里简化处理：计算所有 Patch 对之间的相似度
        patches_flat = patches.view(num_patches, -1)  # [num_patches, C*patch_size*patch_size]
        
        # 归一化
        patches_norm = F.normalize(patches_flat, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(patches_norm, patches_norm.t())  # [num_patches, num_patches]
        
        # 我们希望相邻的 Patch 相似，但不要过度相似（避免过度平滑）
        # 使用稀疏性约束：鼓励 Patch 之间的差异
        # 这里使用 L1 正则化来鼓励稀疏性
        loss_sparse = torch.mean(torch.abs(similarity_matrix - torch.eye(
            num_patches, device=self.device
        )))
        
        return loss_sparse
    
    def compute_patch_smoothness(
        self, 
        patches: torch.Tensor,
        grid_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        计算 Patch 内部的平滑度
        
        Args:
            patches: Patch 张量 [num_patches, C, patch_size, patch_size]
            grid_shape: Patch 网格形状 (num_patches_h, num_patches_w)
            
        Returns:
            平滑度损失（标量）
        """
        num_patches_h, num_patches_w = grid_shape
        
        # 计算每个 Patch 内部的梯度
        total_smoothness = 0.0
        
        for i in range(patches.shape[0]):
            patch = patches[i]  # [C, patch_size, patch_size]
            
            # 计算水平和垂直梯度
            grad_x = patch[:, :, 1:] - patch[:, :, :-1]
            grad_y = patch[:, 1:, :] - patch[:, :-1, :]
            
            # L2 平滑度
            smoothness = torch.mean(grad_x ** 2) + torch.mean(grad_y ** 2)
            total_smoothness += smoothness
        
        return total_smoothness / patches.shape[0]
    
    def forward(self, texture_map: torch.Tensor) -> torch.Tensor:
        """
        计算显式稀疏优化损失
        
        Args:
            texture_map: 纹理贴图 [C, H, W]
            
        Returns:
            损失值（标量）
        """
        # 提取 Patch
        patches, grid_shape = self.extract_patches(texture_map)
        
        # 计算 Patch 一致性损失
        loss_consistency = self.compute_patch_consistency(patches)
        
        # 计算 Patch 平滑度损失
        loss_smoothness = self.compute_patch_smoothness(patches, grid_shape)
        
        # 总损失：平衡一致性和平滑度
        # 一致性损失鼓励 Patch 之间的合理差异
        # 平滑度损失鼓励 Patch 内部的连续性
        total_loss = 0.5 * loss_consistency + 0.5 * loss_smoothness
        
        return total_loss
    
    def optimize_patches(
        self,
        patches: torch.Tensor,
        num_iterations: int = 50,
        lr: float = 0.01
    ) -> torch.Tensor:
        """
        对 Patch 进行显式优化
        
        Args:
            patches: 初始 Patch 张量 [num_patches, C, patch_size, patch_size]
            num_iterations: 优化迭代次数
            lr: 学习率
            
        Returns:
            优化后的 Patch 张量
        """
        patches_opt = patches.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([patches_opt], lr=lr)
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # 计算损失（需要重建纹理）
            # 这里简化处理，直接对 Patch 计算损失
            patches_flat = patches_opt.view(patches_opt.shape[0], -1)
            patches_norm = F.normalize(patches_flat, p=2, dim=1)
            similarity = torch.mm(patches_norm, patches_norm.t())
            
            loss = torch.mean(torch.abs(similarity - torch.eye(
                patches_opt.shape[0], device=patches_opt.device
            )))
            
            loss.backward()
            optimizer.step()
        
        return patches_opt.detach()

