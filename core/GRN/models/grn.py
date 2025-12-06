"""
Geometry Refinement Network (GRN)
轻量级UNet架构，用于修复3D几何中的噪声、孔洞和法线不一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class GRN(nn.Module):
    """
    Geometry Refinement Network
    
    架构：
    - 输入通道：5 (Depth=1, Normal=3, Mask=1)
    - 输出通道：4 (Refined Depth=1, Refined Normal=3)
    - 通道配置：32 → 64 → 128 → 64 → 32
    - 参数量：约0.8M ~ 1.3M
    """
    
    def __init__(self, in_channels=5, out_channels=4):
        super(GRN, self).__init__()
        
        # 编码器（下采样）
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = DoubleConv(128, 128)
        
        # 解码器（上采样）
        # 注意：跳跃连接会连接编码器和解码器特征，所以输入通道数需要匹配
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 64, 64)  # 128(enc3) + 64(up3) = 192 -> 64
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 32, 32)  # 64(enc2) + 32(up2) = 96 -> 32
        
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32 + 32, 32)  # 32(enc1) + 32(up1) = 64 -> 32
        
        # 输出层
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        x1 = self.pool1(enc1)
        
        enc2 = self.enc2(x1)
        x2 = self.pool2(enc2)
        
        enc3 = self.enc3(x2)
        x3 = self.pool3(enc3)
        
        # 瓶颈层
        bottleneck = self.bottleneck(x3)
        
        # 解码路径（带跳跃连接）
        up3 = self.up3(bottleneck)
        up3 = F.interpolate(up3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        
        up2 = self.up2(dec3)
        up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.up1(dec2)
        up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        # 输出
        output = self.final_conv(dec1)
        
        return output
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_grn_model(in_channels=5, out_channels=4):
    """创建GRN模型的工厂函数"""
    model = GRN(in_channels=in_channels, out_channels=out_channels)
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_grn_model()
    print(f"模型参数量: {model.count_parameters() / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(1, 5, 256, 256)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

