"""
纹理一致性增强 (Texture Consistency Enhancement, TCE) 模块

解决"多头问题"及纹理接缝的核心模块，包含三个核心策略：
1. CLIP 语义一致性
2. Warp Consistency（光流法）
3. 显式稀疏优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from clip_consistency import CLIPConsistencyLoss
from warp_consistency import WarpConsistencyLoss
from sparse_optimization import SparsePatchOptimizer


class TCEModule(nn.Module):
    """
    纹理一致性增强模块
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        lambda_sem: float = 1.0,
        lambda_warp: float = 1.0,
        lambda_sparse: float = 1.0,
        patch_size: int = 16,
        device: str = "cuda"
    ):
        """
        初始化 TCE 模块
        
        Args:
            clip_model_name: CLIP 模型名称
            lambda_sem: CLIP 语义一致性损失权重
            lambda_warp: Warp Consistency 损失权重
            lambda_sparse: 稀疏优化损失权重
            patch_size: Patch 大小（用于稀疏优化）
            device: 计算设备
        """
        super().__init__()
        
        self.lambda_sem = lambda_sem
        self.lambda_warp = lambda_warp
        self.lambda_sparse = lambda_sparse
        self.patch_size = patch_size
        self.device = device
        
        # 初始化三个核心组件
        self.clip_consistency = CLIPConsistencyLoss(
            model_name=clip_model_name,
            device=device
        )
        
        self.warp_consistency = WarpConsistencyLoss(device=device)
        
        self.sparse_optimizer = SparsePatchOptimizer(
            patch_size=patch_size,
            device=device
        )
    
    def forward(
        self,
        rendered_images: torch.Tensor,
        text_prompt: str,
        texture_map: torch.Tensor,
        view_pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        optical_flows: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，计算所有损失
        
        Args:
            rendered_images: 渲染图像 [B, C, H, W]
            text_prompt: 文本提示
            texture_map: 纹理贴图 [C, H, W]
            view_pairs: 相邻视角对 (view1, view2)，可选
            optical_flows: 预计算的光流 [B, 2, H, W]，可选
            
        Returns:
            包含各项损失的字典
        """
        losses = {}
        
        # 1. CLIP 语义一致性损失
        loss_sem = self.clip_consistency(rendered_images, text_prompt)
        losses['clip_semantic'] = loss_sem
        
        # 2. Warp Consistency 损失
        if view_pairs is not None:
            loss_warp = self.warp_consistency(
                texture_map, 
                view_pairs, 
                optical_flows
            )
            losses['warp_consistency'] = loss_warp
        else:
            losses['warp_consistency'] = torch.tensor(0.0, device=self.device)
        
        # 3. 显式稀疏优化损失
        loss_sparse = self.sparse_optimizer(texture_map)
        losses['sparse_optimization'] = loss_sparse
        
        # 总损失
        total_loss = (
            self.lambda_sem * loss_sem +
            self.lambda_warp * losses['warp_consistency'] +
            self.lambda_sparse * loss_sparse
        )
        losses['total'] = total_loss
        
        return losses
    
    def optimize_texture(
        self,
        texture_map: torch.Tensor,
        rendered_images: torch.Tensor,
        text_prompt: str,
        view_pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_iterations: int = 100,
        lr: float = 0.01
    ) -> torch.Tensor:
        """
        对纹理贴图进行显式优化
        
        Args:
            texture_map: 初始纹理贴图 [C, H, W]
            rendered_images: 渲染图像 [B, C, H, W]
            text_prompt: 文本提示
            view_pairs: 相邻视角对，可选
            num_iterations: 优化迭代次数
            lr: 学习率
            
        Returns:
            优化后的纹理贴图
        """
        texture = texture_map.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([texture], lr=lr)
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # 计算损失
            losses = self.forward(
                rendered_images=rendered_images,
                text_prompt=text_prompt,
                texture_map=texture,
                view_pairs=view_pairs
            )
            
            # 反向传播
            losses['total'].backward()
            optimizer.step()
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, "
                      f"Total Loss: {losses['total'].item():.4f}, "
                      f"CLIP: {losses['clip_semantic'].item():.4f}, "
                      f"Warp: {losses['warp_consistency'].item():.4f}, "
                      f"Sparse: {losses['sparse_optimization'].item():.4f}")
        
        return texture.detach()

