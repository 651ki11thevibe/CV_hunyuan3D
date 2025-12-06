"""
TCE 模块使用示例

演示如何使用纹理一致性增强模块进行纹理优化
"""

import torch
import torch.nn.functional as F
from tce_module import TCEModule


def example_basic_usage():
    """基本使用示例"""
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 初始化 TCE 模块
    tce = TCEModule(
        clip_model_name="ViT-B/32",
        lambda_sem=1.0,      # CLIP 语义一致性权重
        lambda_warp=1.0,     # Warp Consistency 权重
        lambda_sparse=0.5,   # 稀疏优化权重
        patch_size=16,
        device=device
    )
    
    # 创建模拟数据
    batch_size = 4
    height, width = 256, 256
    channels = 3
    
    # 渲染图像（模拟从不同视角渲染的结果）
    rendered_images = torch.rand(batch_size, channels, height, width).to(device)
    
    # 纹理贴图
    texture_map = torch.rand(channels, height, width).to(device)
    
    # 文本提示
    text_prompt = "a beautiful landscape with mountains and trees"
    
    # 相邻视角对（可选）
    view1 = torch.rand(channels, height, width).to(device)
    view2 = torch.rand(channels, height, width).to(device)
    view_pairs = (view1, view2)
    
    # 前向传播，计算损失
    losses = tce(
        rendered_images=rendered_images,
        text_prompt=text_prompt,
        texture_map=texture_map,
        view_pairs=view_pairs
    )
    
    print("\n损失值:")
    print(f"  CLIP 语义一致性: {losses['clip_semantic'].item():.4f}")
    print(f"  Warp Consistency: {losses['warp_consistency'].item():.4f}")
    print(f"  稀疏优化: {losses['sparse_optimization'].item():.4f}")
    print(f"  总损失: {losses['total'].item():.4f}")
    
    return losses


def example_texture_optimization():
    """纹理优化示例"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 初始化 TCE 模块
    tce = TCEModule(
        clip_model_name="ViT-B/32",
        lambda_sem=1.0,
        lambda_warp=1.0,
        lambda_sparse=0.5,
        patch_size=16,
        device=device
    )
    
    # 创建初始纹理
    channels, height, width = 3, 256, 256
    initial_texture = torch.rand(channels, height, width).to(device)
    
    # 渲染图像
    batch_size = 4
    rendered_images = torch.rand(batch_size, channels, height, width).to(device)
    
    # 文本提示
    text_prompt = "a detailed texture with consistent patterns"
    
    # 视角对
    view1 = torch.rand(channels, height, width).to(device)
    view2 = torch.rand(channels, height, width).to(device)
    view_pairs = (view1, view2)
    
    print("\n开始纹理优化...")
    
    # 优化纹理
    optimized_texture = tce.optimize_texture(
        texture_map=initial_texture,
        rendered_images=rendered_images,
        text_prompt=text_prompt,
        view_pairs=view_pairs,
        num_iterations=50,
        lr=0.01
    )
    
    print("\n优化完成！")
    print(f"初始纹理形状: {initial_texture.shape}")
    print(f"优化后纹理形状: {optimized_texture.shape}")
    
    return optimized_texture


def example_individual_components():
    """单独使用各个组件的示例"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. CLIP 语义一致性
    from clip_consistency import CLIPConsistencyLoss
    
    clip_loss = CLIPConsistencyLoss(device=device)
    images = torch.rand(2, 3, 256, 256).to(device)
    text = "a beautiful scene"
    loss_clip = clip_loss(images, text)
    print(f"\nCLIP 损失: {loss_clip.item():.4f}")
    
    # 2. Warp Consistency
    from warp_consistency import WarpConsistencyLoss
    
    warp_loss = WarpConsistencyLoss(device=device)
    texture = torch.rand(3, 256, 256).to(device)
    view1 = torch.rand(3, 256, 256).to(device)
    view2 = torch.rand(3, 256, 256).to(device)
    loss_warp = warp_loss(texture, (view1, view2))
    print(f"Warp 损失: {loss_warp.item():.4f}")
    
    # 3. 稀疏优化
    from sparse_optimization import SparsePatchOptimizer
    
    sparse_opt = SparsePatchOptimizer(patch_size=16, device=device)
    loss_sparse = sparse_opt(texture)
    print(f"稀疏优化损失: {loss_sparse.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("TCE 模块使用示例")
    print("=" * 60)
    
    # 基本使用
    example_basic_usage()
    
    # 纹理优化
    # example_texture_optimization()  # 取消注释以运行优化（需要较长时间）
    
    # 单独组件
    example_individual_components()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)

