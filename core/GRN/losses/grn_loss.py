"""
GRN损失函数
根据论文公式：Lgeo = Lrec + λsm Lsmooth + λnc Lnormal + λlap Llap

其中：
- Lrec: L1重建损失
- Lsmooth: 局部平滑度约束
- Lnormal: 法线一致性损失
- Llap: 拉普拉斯平滑项 ||∆v − Lap(∆v)||²₂
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRNLoss(nn.Module):
    """
    GRN损失函数组合
    
    损失函数公式：Lgeo = Lrec + λsm Lsmooth + λnc Lnormal + λlap Llap
    
    包括：
    1. Lrec: L1重建损失
    2. Lsmooth: 局部平滑度约束（Total Variation）
    3. Lnormal: 法线一致性损失
    4. Llap: 拉普拉斯平滑项
    """
    
    def __init__(
        self,
        rec_weight=1.0,
        smooth_weight=0.1,
        normal_weight=2.0,
        laplacian_weight=0.05
    ):
        super(GRNLoss, self).__init__()
        self.rec_weight = rec_weight  # λrec (通常为1.0)
        self.smooth_weight = smooth_weight  # λsm
        self.normal_weight = normal_weight  # λnc
        self.laplacian_weight = laplacian_weight  # λlap
    
    def reconstruction_loss(self, pred_depth: torch.Tensor, target_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Lrec: L1重建损失
        
        计算预测深度和目标深度之间的L1距离
        """
        # 只在有效区域计算损失
        # mask形状: [B, H, W]，需要扩展到 [B, 1, H, W] 以匹配深度图
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        valid_mask = mask > 0.5
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_depth.device)
        
        pred_depth_masked = pred_depth[valid_mask]
        target_depth_masked = target_depth[valid_mask]
        
        # L1损失
        l1_loss = F.l1_loss(pred_depth_masked, target_depth_masked)
        
        return l1_loss
    
    def normal_consistency_loss(self, pred_normal: torch.Tensor, target_normal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Lnormal: 法线一致性损失
        
        保证预测法线与目标法线的一致性
        """
        # 归一化法线到[-1, 1]
        pred_normal = pred_normal * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        target_normal = target_normal * 2.0 - 1.0
        
        # 只在有效区域计算损失
        # mask形状: [B, H, W]，需要扩展到 [B, 1, H, W] 以匹配法线图
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        valid_mask = mask > 0.5
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_normal.device)
        
        # 展平空间维度
        B, C, H, W = pred_normal.shape
        # 扩展mask以匹配法线图的通道数
        valid_mask_expanded = valid_mask.expand(-1, C, -1, -1)
        pred_flat = pred_normal.permute(0, 2, 3, 1).reshape(-1, C)[valid_mask_expanded.permute(0, 2, 3, 1).reshape(-1, C).any(dim=1)]
        target_flat = target_normal.permute(0, 2, 3, 1).reshape(-1, C)[valid_mask_expanded.permute(0, 2, 3, 1).reshape(-1, C).any(dim=1)]
        
        if pred_flat.numel() == 0:
            return torch.tensor(0.0, device=pred_normal.device)
        
        # 归一化
        pred_flat = F.normalize(pred_flat, p=2, dim=1)
        target_flat = F.normalize(target_flat, p=2, dim=1)
        
        # 余弦相似度损失（1 - 余弦相似度）
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1)
        normal_loss = (1 - cosine_sim).mean()
        
        return normal_loss
    
    def smoothness_loss(self, pred_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Lsmooth: 局部平滑度约束
        
        使用Total Variation约束局部平滑度
        """
        # mask形状: [B, H, W]，需要扩展到 [B, 1, H, W] 以匹配深度图
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        valid_mask = mask > 0.5
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_depth.device)
        
        # 计算梯度
        grad_x = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])
        grad_y = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])
        
        # 只在有效区域计算
        mask_x = valid_mask[:, :, :, :-1] & valid_mask[:, :, :, 1:]
        mask_y = valid_mask[:, :, :-1, :] & valid_mask[:, :, 1:, :]
        
        smoothness = 0.0
        if mask_x.any():
            smoothness += grad_x[mask_x].mean()
        if mask_y.any():
            smoothness += grad_y[mask_y].mean()
        
        return smoothness
    
    def laplacian_smooth_loss(self, pred_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Llap: 拉普拉斯平滑项
        
        公式：Llap = ||∆v − Lap(∆v)||²₂
        
        其中：
        - ∆v: 深度变化（可以理解为深度值的局部变化）
        - Lap(∆v): 深度变化的拉普拉斯平滑
        
        对于2D图像，拉普拉斯算子定义为：
        Lap(f) = f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4*f(x,y)
        
        这里我们将深度值本身视为 ∆v，计算其拉普拉斯平滑，然后计算差异
        """
        # mask形状: [B, H, W]，需要扩展到 [B, 1, H, W] 以匹配深度图
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        valid_mask = mask > 0.5
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_depth.device)
        
        # 拉普拉斯核（4邻域）
        laplacian_kernel = torch.tensor([[0, 1, 0], 
                                        [1, -4, 1], 
                                        [0, 1, 0]], 
                                       dtype=pred_depth.dtype, device=pred_depth.device).view(1, 1, 3, 3)
        
        # 计算深度的拉普拉斯 Lap(∆v)
        lap_depth = F.conv2d(pred_depth, laplacian_kernel, padding=1)
        
        # 计算深度与其拉普拉斯平滑的差异
        # ∆v - Lap(∆v)，其中 ∆v 可以理解为深度值本身
        # 为了更符合网格优化的语义，我们计算深度变化与其拉普拉斯平滑的差异
        delta_v = pred_depth  # 深度值作为 ∆v
        
        # 计算 ||∆v − Lap(∆v)||²₂
        # 这里我们计算深度值与其拉普拉斯平滑版本的差异
        diff = delta_v - lap_depth
        
        # 只在有效区域计算
        diff_masked = diff[valid_mask]
        
        if diff_masked.numel() > 0:
            laplacian_loss = (diff_masked ** 2).mean()
        else:
            laplacian_loss = torch.tensor(0.0, device=pred_depth.device)
        
        return laplacian_loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> dict:
        """
        计算总损失
        
        损失函数公式：Lgeo = Lrec + λsm Lsmooth + λnc Lnormal + λlap Llap
        
        Args:
            pred: 预测输出 [B, 4, H, W] (Depth=1, Normal=3)
            target: 目标输出 [B, 4, H, W] (Depth=1, Normal=3)
            mask: 有效掩码 [B, H, W]
        
        Returns:
            损失字典，包含各项损失和总损失
        """
        # 分离深度和法线
        pred_depth = pred[:, 0:1, :, :]
        pred_normal = pred[:, 1:4, :, :]
        
        target_depth = target[:, 0:1, :, :]
        target_normal = target[:, 1:4, :, :]
        
        # 计算各项损失
        Lrec = self.reconstruction_loss(pred_depth, target_depth, mask)
        Lsmooth = self.smoothness_loss(pred_depth, mask)
        Lnormal = self.normal_consistency_loss(pred_normal, target_normal, mask)
        Llap = self.laplacian_smooth_loss(pred_depth, mask)
        
        # 总损失：Lgeo = Lrec + λsm Lsmooth + λnc Lnormal + λlap Llap
        Lgeo = (
            self.rec_weight * Lrec +
            self.smooth_weight * Lsmooth +
            self.normal_weight * Lnormal +
            self.laplacian_weight * Llap
        )
        
        return {
            'total_loss': Lgeo,  # Lgeo
            'Lrec': Lrec,
            'Lsmooth': Lsmooth,
            'Lnormal': Lnormal,
            'Llap': Llap
        }


if __name__ == "__main__":
    # 测试损失函数
    loss_fn = GRNLoss(
        rec_weight=1.0,
        smooth_weight=0.1,
        normal_weight=2.0,
        laplacian_weight=0.05
    )
    
    B, H, W = 2, 256, 256
    pred = torch.randn(B, 4, H, W)
    target = torch.randn(B, 4, H, W)
    mask = torch.ones(B, H, W)
    
    losses = loss_fn(pred, target, mask)
    print("损失值 (Lgeo = Lrec + λsm Lsmooth + λnc Lnormal + λlap Llap):")
    print(f"  Lgeo (总损失): {losses['total_loss'].item():.4f}")
    print(f"  Lrec (重建损失): {losses['Lrec'].item():.4f}")
    print(f"  Lsmooth (平滑损失): {losses['Lsmooth'].item():.4f}")
    print(f"  Lnormal (法线损失): {losses['Lnormal'].item():.4f}")
    print(f"  Llap (拉普拉斯损失): {losses['Llap'].item():.4f}")

