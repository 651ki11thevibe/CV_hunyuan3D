"""
CLIP 语义一致性损失模块

实现公式 (3): L_sem = 1 - sim(CLIP(I_render), CLIP(T_prompt))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Union


class CLIPConsistencyLoss(nn.Module):
    """
    CLIP 语义一致性损失
    
    约束从任意视角渲染的图像 I_render 与文本提示 T_prompt 
    在 CLIP 特征空间的一致性
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda"
    ):
        """
        初始化 CLIP 一致性损失模块
        
        Args:
            model_name: CLIP 模型名称
            device: 计算设备
        """
        super().__init__()
        
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        # 冻结 CLIP 模型参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像到 CLIP 特征空间
        
        Args:
            images: 图像张量 [B, C, H, W]，值域 [0, 1]
            
        Returns:
            CLIP 图像特征 [B, feature_dim]
        """
        # 将图像从 [0, 1] 归一化到 CLIP 期望的输入范围
        # CLIP 期望输入为 [0, 1] 的 RGB 图像
        images = torch.clamp(images, 0, 1)
        
        # 如果图像尺寸不符合 CLIP 输入要求，需要调整
        # CLIP 通常期望 224x224
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(
                images, 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 编码图像
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        编码文本到 CLIP 特征空间
        
        Args:
            text: 文本提示字符串
            
        Returns:
            CLIP 文本特征 [1, feature_dim]
        """
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def compute_similarity(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 CLIP 特征之间的相似度
        
        Args:
            image_features: 图像特征 [B, feature_dim]
            text_features: 文本特征 [1, feature_dim]
            
        Returns:
            相似度分数 [B]
        """
        # 计算余弦相似度
        similarity = (image_features * text_features).sum(dim=-1)
        return similarity
    
    def forward(
        self, 
        rendered_images: torch.Tensor, 
        text_prompt: str
    ) -> torch.Tensor:
        """
        计算 CLIP 语义一致性损失
        
        公式: L_sem = 1 - sim(CLIP(I_render), CLIP(T_prompt))
        
        Args:
            rendered_images: 渲染图像 [B, C, H, W]，值域 [0, 1]
            text_prompt: 文本提示字符串
            
        Returns:
            损失值（标量）
        """
        # 编码图像和文本
        image_features = self.encode_image(rendered_images)
        text_features = self.encode_text(text_prompt)
        
        # 计算相似度
        similarity = self.compute_similarity(image_features, text_features)
        
        # 计算损失：1 - similarity
        # 对批次取平均
        loss = 1.0 - similarity.mean()
        
        return loss

