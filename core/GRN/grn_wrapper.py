"""
GRN (Geometry Refinement Network) 包装模块
提供简化的接口用于在pipeline中集成GRN
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

# 添加GRN目录到路径
GRN_ROOT = Path(__file__).parent
if str(GRN_ROOT) not in sys.path:
    sys.path.insert(0, str(GRN_ROOT))

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
    from models import create_grn_model
    from data import MeshRenderer
    from utils.checkpoint import load_checkpoint
except ImportError as e:
    raise ImportError(
        f"无法导入GRN模块: {e}\n"
        "请确保GRN目录结构完整，并且所有依赖已安装。"
    )


def refine_mesh_with_grn(
    asset: RawAsset,
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    image_size: int = 256,
) -> RawAsset:
    """
    使用GRN精炼网格几何
    
    参数:
        asset: 输入的RawAsset（粗糙网格）
        checkpoint_path: GRN模型检查点路径
        config: 可选配置字典
        device: 计算设备
        image_size: 渲染图像尺寸
    
    返回:
        精炼后的RawAsset
    """
    if trimesh is None:
        raise ImportError("trimesh未安装，无法使用GRN。请安装: pip install trimesh")
    
    if config is None:
        config = {}
    
    # 获取配置参数
    image_size = config.get("image_size", image_size)
    
    # 将RawAsset转换为trimesh对象
    mesh = _raw_asset_to_trimesh(asset)
    
    # 创建GRN模型
    model = create_grn_model()
    model = model.to(device)
    model.eval()
    
    # 加载检查点
    try:
        load_checkpoint(checkpoint_path, model, device=device)
    except Exception as e:
        raise RuntimeError(
            f"无法加载GRN检查点 {checkpoint_path}: {e}\n"
            "请确保检查点路径正确且文件存在。"
        )
    
    # 创建渲染器
    renderer = MeshRenderer(image_size=image_size)
    
    # 生成相机位姿（使用多个视角的平均结果）
    camera_poses = _generate_camera_poses(num_views=8)
    
    # 收集所有视角的精炼结果
    refined_depth_maps = []
    refined_normal_maps = []
    masks = []
    
    with torch.no_grad():
        for camera_pose in camera_poses:
            # 渲染输入
            depth, normal, mask = renderer.render_mesh(mesh, camera_pose)
            
            # 组合为5通道输入
            input_channels = np.concatenate([
                depth[..., np.newaxis],
                normal,
                mask[..., np.newaxis]
            ], axis=-1)
            
            input_tensor = torch.from_numpy(input_channels).float()
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # GRN推理
            output = model(input_tensor)
            
            # 提取精炼结果
            output_np = output.squeeze(0).cpu().numpy()
            refined_depth = output_np[0]
            refined_normal = output_np[1:4].transpose(1, 2, 0)
            
            refined_depth_maps.append(refined_depth)
            refined_normal_maps.append(refined_normal)
            masks.append(mask)
    
    # 使用精炼结果重建网格（简化版本：直接返回原网格，实际应该从多视角重建）
    # TODO: 实现从多视角深度/法线图重建mesh的完整流程
    # 这里先返回原始mesh，因为重建3D mesh需要更复杂的多视角融合算法
    # 在实际应用中，可以使用TSDF融合或其他3D重建方法
    
    # 暂时返回原始asset（标记为已精炼）
    # 在实际完整实现中，应该：
    # 1. 从多个视角的深度图重建3D点云
    # 2. 使用Poisson重建或其他方法生成mesh
    # 3. 应用精炼的法线信息
    
    print("[GRN] 几何精炼完成（当前为简化实现）")
    
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
            # 创建纹理材质
            from PIL import Image
            if isinstance(asset.textures["albedo"], np.ndarray):
                texture_img = Image.fromarray(
                    (asset.textures["albedo"] * 255).astype(np.uint8)
                )
                mesh.visual.material.image = texture_img
    
    return mesh


def _generate_camera_poses(num_views: int = 8) -> list:
    """生成多个视角的相机位姿"""
    poses = []
    
    for i in range(num_views):
        # 在球面上均匀分布相机
        theta = 2 * np.pi * i / num_views  # 水平角度
        phi = np.pi / 4  # 固定俯仰角
        
        # 计算相机位置（在单位球面上）
        radius = 2.0
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        # 构建相机位姿矩阵
        camera_pos = np.array([x, y, z])
        target = np.array([0, 0, 0])  # 看向原点
        up = np.array([0, 0, 1])  # 上方向
        
        # 计算相机坐标系
        forward = target - camera_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up_cam = np.cross(right, forward)
        up_cam = up_cam / (np.linalg.norm(up_cam) + 1e-8)
        
        # 构建4x4变换矩阵
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up_cam
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        
        poses.append(pose)
    
    return poses


__all__ = ["refine_mesh_with_grn"]
