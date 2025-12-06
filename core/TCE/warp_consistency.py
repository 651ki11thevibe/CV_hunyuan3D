"""
Warp Consistency 损失模块

利用光流法约束相邻视角的纹理像素对应关系，强迫纹理在空间上连续
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class WarpConsistencyLoss(nn.Module):
    """
    Warp Consistency 损失
    
    利用光流法约束相邻视角的纹理像素对应关系
    """
    
    def __init__(self, device: str = "cuda"):
        """
        初始化 Warp Consistency 损失模块
        
        Args:
            device: 计算设备
        """
        super().__init__()
        self.device = device
    
    def compute_optical_flow(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两张图像之间的光流
        
        这里使用简化的光流计算方法，实际应用中可以使用
        RAFT、FlowNet 等更精确的光流估计网络
        
        Args:
            img1: 第一张图像 [C, H, W]
            img2: 第二张图像 [C, H, W]
            
        Returns:
            光流场 [2, H, W]，其中 [0, :, :] 是 x 方向，[1, :, :] 是 y 方向
        """
        # 转换为灰度图
        if img1.shape[0] == 3:
            gray1 = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
            gray2 = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
        else:
            gray1 = img1[0]
            gray2 = img2[0]
        
        gray1 = gray1.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        gray2 = gray2.unsqueeze(0).unsqueeze(0)
        
        # 使用 Sobel 算子计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        Ix = F.conv2d(gray1, sobel_x, padding=1)
        Iy = F.conv2d(gray1, sobel_y, padding=1)
        It = gray2 - gray1
        
        # Lucas-Kanade 方法计算光流（简化版）
        # 使用局部窗口计算
        window_size = 5
        padding = window_size // 2
        
        Ix_padded = F.pad(Ix, (padding, padding, padding, padding), mode='reflect')
        Iy_padded = F.pad(Iy, (padding, padding, padding, padding), mode='reflect')
        It_padded = F.pad(It, (padding, padding, padding, padding), mode='reflect')
        
        H, W = Ix.shape[-2], Ix.shape[-1]
        flow_x = torch.zeros(H, W, device=self.device)
        flow_y = torch.zeros(H, W, device=self.device)
        
        for i in range(padding, H + padding):
            for j in range(padding, W + padding):
                Ix_window = Ix_padded[0, 0, i-padding:i+padding+1, j-padding:j+padding+1].flatten()
                Iy_window = Iy_padded[0, 0, i-padding:i+padding+1, j-padding:j+padding+1].flatten()
                It_window = It_padded[0, 0, i-padding:i+padding+1, j-padding:j+padding+1].flatten()
                
                # 构建线性系统 A * [u, v]^T = b
                A = torch.stack([
                    (Ix_window * Ix_window).sum(),
                    (Ix_window * Iy_window).sum(),
                    (Ix_window * Iy_window).sum(),
                    (Iy_window * Iy_window).sum()
                ]).view(2, 2)
                
                b = -torch.stack([
                    (Ix_window * It_window).sum(),
                    (Iy_window * It_window).sum()
                ])
                
                # 求解光流
                try:
                    flow = torch.linalg.solve(A, b)
                    flow_x[i-padding, j-padding] = flow[0]
                    flow_y[i-padding, j-padding] = flow[1]
                except:
                    flow_x[i-padding, j-padding] = 0
                    flow_y[i-padding, j-padding] = 0
        
        flow = torch.stack([flow_x, flow_y])  # [2, H, W]
        return flow
    
    def warp_texture(
        self, 
        texture: torch.Tensor, 
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        根据光流对纹理进行变形
        
        Args:
            texture: 纹理贴图 [C, H, W]
            flow: 光流场 [2, H, W]
            
        Returns:
            变形后的纹理 [C, H, W]
        """
        C, H, W = texture.shape
        texture = texture.unsqueeze(0)  # [1, C, H, W]
        
        # 创建坐标网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 应用光流
        grid_x = grid_x + flow[0]
        grid_y = grid_y + flow[1]
        
        # 归一化到 [-1, 1]
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        
        # 双线性插值
        warped = F.grid_sample(
            texture, 
            grid.permute(0, 2, 3, 1), 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        )
        
        return warped.squeeze(0)  # [C, H, W]
    
    def forward(
        self,
        texture_map: torch.Tensor,
        view_pairs: Tuple[torch.Tensor, torch.Tensor],
        optical_flows: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 Warp Consistency 损失
        
        Args:
            texture_map: 纹理贴图 [C, H, W]
            view_pairs: 相邻视角对 (view1, view2)，每个都是 [C, H, W]
            optical_flows: 预计算的光流 [2, H, W]，可选
            
        Returns:
            损失值（标量）
        """
        view1, view2 = view_pairs
        
        # 如果没有提供光流，则计算
        if optical_flows is None:
            flow = self.compute_optical_flow(view1, view2)
        else:
            flow = optical_flows
        
        # 对纹理进行变形
        texture_warped = self.warp_texture(texture_map, flow)
        
        # 计算原始纹理和变形后纹理的一致性
        # 这里使用 L1 损失
        loss = F.l1_loss(texture_map, texture_warped)
        
        return loss

