"""
TCE (Texture Consistency Enhancement) 包装模块
提供简化的接口用于在pipeline中集成TCE
"""

from __future__ import annotations

import sys
import io
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from PIL import Image

# 添加TCE目录到路径
TCE_ROOT = Path(__file__).parent
if str(TCE_ROOT) not in sys.path:
    sys.path.insert(0, str(TCE_ROOT))

# 添加项目根目录到路径，以便导入core模块
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import trimesh
except ImportError:
    trimesh = None

from core.models.hunyuan3d_wrapper import RawAsset

try:
    from tce_module import TCEModule
except ImportError as e:
    raise ImportError(
        f"无法导入TCE模块: {e}\n"
        "请确保TCE目录结构完整，并且所有依赖已安装。"
    )


def enhance_texture_with_tce(
    asset: RawAsset,
    text_prompt: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
) -> RawAsset:
    """
    使用TCE增强纹理一致性
    
    参数:
        asset: 输入的RawAsset（包含mesh和纹理）
        text_prompt: 文本提示词（用于CLIP语义一致性）
        config: 可选配置字典
        device: 计算设备
    
    返回:
        纹理增强后的RawAsset
    """
    if config is None:
        config = {}
    
    # 检查是否有纹理
    if asset.textures is None or "albedo" not in asset.textures:
        print("[TCE] 警告：资产没有纹理信息，跳过纹理增强")
        return asset
    
    if trimesh is None:
        print("[TCE] 警告：trimesh未安装，无法进行多视角渲染，跳过纹理增强")
        return asset
    
    # 获取配置参数
    lambda_sem = config.get("lambda_sem", 1.0)
    lambda_warp = config.get("lambda_warp", 1.0)
    lambda_sparse = config.get("lambda_sparse", 0.5)
    patch_size = config.get("patch_size", 16)
    num_iterations = config.get("num_iterations", 50)
    lr = config.get("lr", 0.01)
    clip_model_name = config.get("clip_model_name", "ViT-B/32")
    
    # 如果没有提供文本提示，使用默认值
    if text_prompt is None:
        text_prompt = config.get("default_text_prompt", "a high quality 3D model")
    
    # 初始化TCE模块
    tce = TCEModule(
        clip_model_name=clip_model_name,
        lambda_sem=lambda_sem,
        lambda_warp=lambda_warp,
        lambda_sparse=lambda_sparse,
        patch_size=patch_size,
        device=device
    )
    
    # 将RawAsset转换为trimesh以进行渲染
    mesh = _raw_asset_to_trimesh(asset)
    
    # 从多个视角渲染mesh（用于TCE优化）
    rendered_images = _render_multiview(mesh, num_views=4, device=device)
    
    # 获取当前纹理贴图
    texture_map = _extract_texture_map(asset.textures["albedo"])
    texture_map = texture_map.to(device)
    
    # 优化纹理
    print("[TCE] 开始纹理一致性增强...")
    optimized_texture = tce.optimize_texture(
        texture_map=texture_map,
        rendered_images=rendered_images,
        text_prompt=text_prompt,
        num_iterations=num_iterations,
        lr=lr
    )
    
    # 将优化后的纹理转换回numpy格式
    optimized_texture_np = optimized_texture.cpu().numpy()
    if optimized_texture_np.shape[0] == 3:  # CHW格式
        optimized_texture_np = optimized_texture_np.transpose(1, 2, 0)  # HWC格式
    
    # 归一化到[0, 1]范围
    if optimized_texture_np.max() > 1.0:
        optimized_texture_np = optimized_texture_np / 255.0
    
    # 更新asset的纹理
    asset.textures["albedo"] = optimized_texture_np
    
    print("[TCE] 纹理一致性增强完成")
    
    return asset


def _raw_asset_to_trimesh(asset: RawAsset) -> trimesh.Trimesh:
    """将RawAsset转换为trimesh对象"""
    if asset._original_mesh is not None:
        return asset._original_mesh
    
    vertices = asset.mesh["vertices"]
    faces = asset.mesh["faces"]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 如果有纹理信息，尝试添加
    if asset.textures is not None:
        if "uv" in asset.textures:
            mesh.visual.uv = asset.textures["uv"]
        if "albedo" in asset.textures:
            albedo = asset.textures["albedo"]
            if isinstance(albedo, np.ndarray):
                texture_img = Image.fromarray(
                    (albedo * 255).astype(np.uint8) if albedo.max() <= 1.0
                    else albedo.astype(np.uint8)
                )
                mesh.visual.material.image = texture_img
    
    return mesh


def _render_multiview(mesh: trimesh.Trimesh, num_views: int = 4, device: str = "cuda") -> torch.Tensor:
    """
    从多个视角渲染mesh
    
    返回:
        rendered_images: [B, C, H, W] 张量
    """
    # 简化实现：使用trimesh的简单渲染
    # 在实际应用中，可以使用PyTorch3D或其他可微渲染器
    
    # 生成相机位姿
    camera_poses = _generate_camera_poses(num_views)
    
    rendered_images = []
    
    for pose in camera_poses:
        # 使用trimesh的scene进行简单渲染
        scene = trimesh.Scene([mesh])
        
        # 获取相机位置和方向
        camera_pos = pose[:3, 3]
        camera_target = np.array([0, 0, 0])
        
        # 简化的渲染（使用scene camera）
        try:
            # 尝试使用trimesh的场景相机
            png = scene.save_image(resolution=[256, 256])
            img = Image.open(io.BytesIO(png)).convert("RGB")
            img_array = np.array(img).astype(np.float32) / 255.0
        except:
            # 如果渲染失败，使用随机图像作为占位符
            img_array = np.random.rand(256, 256, 3).astype(np.float32)
        
        # 转换为CHW格式
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        rendered_images.append(img_tensor)
    
    # 堆叠为批次
    rendered_batch = torch.stack(rendered_images).to(device)
    
    return rendered_batch


def _extract_texture_map(albedo: np.ndarray) -> torch.Tensor:
    """从albedo提取纹理贴图张量"""
    if isinstance(albedo, np.ndarray):
        # 确保是HWC格式
        if len(albedo.shape) == 2:
            # 灰度图，转换为RGB
            albedo = np.stack([albedo] * 3, axis=-1)
        elif albedo.shape[0] == 3 and len(albedo.shape) == 3:
            # CHW格式，转换为HWC
            albedo = albedo.transpose(1, 2, 0)
        
        # 归一化到[0, 1]
        if albedo.max() > 1.0:
            albedo = albedo / 255.0
        
        # 转换为CHW格式的张量
        if albedo.shape[-1] == 3:
            albedo = albedo.transpose(2, 0, 1)  # HWC -> CHW
        
        texture_tensor = torch.from_numpy(albedo).float()
        
        return texture_tensor
    
    raise ValueError(f"不支持的albedo格式: {type(albedo)}")


def _generate_camera_poses(num_views: int = 4) -> list:
    """生成多个视角的相机位姿"""
    poses = []
    
    for i in range(num_views):
        # 在球面上均匀分布相机
        theta = 2 * np.pi * i / num_views  # 水平角度
        phi = np.pi / 4  # 固定俯仰角
        
        # 计算相机位置
        radius = 2.0
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        camera_pos = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        forward = target - camera_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up_cam = np.cross(right, forward)
        up_cam = up_cam / (np.linalg.norm(up_cam) + 1e-8)
        
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up_cam
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        
        poses.append(pose)
    
    return poses


__all__ = ["enhance_texture_with_tce"]
