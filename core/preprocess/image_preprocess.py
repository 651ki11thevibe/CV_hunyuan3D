"""
图像预处理工具模块。

本模块面向单张 2D 图像的轻量预处理，并预留若干扩展点，用于：
- 背景移除
- 边缘图提取
- 深度估计
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps

# 尝试导入 BackgroundRemover（Hunyuan3D 官方提供的背景移除工具）
try:
    from hy3dgen.rembg import BackgroundRemover
    HAS_BACKGROUND_REMOVER = True
except ImportError:
    HAS_BACKGROUND_REMOVER = False
    BackgroundRemover = None  # type: ignore


def resize_and_normalize(image: Image.Image, size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    对单张图像进行基础缩放与规范化。

    参数:
        image: 输入 PIL 图像。
        size: 目标尺寸 (width, height)。

    返回:
        已缩放、RGB 模式的图像，可直接送入主干模型。
    """
    image = image.convert("RGB")
    image = ImageOps.fit(image, size, method=Image.BICUBIC)
    return image


def compute_edge_map(image: Image.Image) -> np.ndarray:
    """
    基于边缘的预处理占位实现。

    返回:
        一个非常简单的梯度幅值近似（二维数组）。

    TODO:
        替换为更可靠的边缘检测算法（例如 Canny），或
        可学习的结构/边缘提取器，用于为 3D 生成器提供先验。
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    # Simple finite differences as a placeholder.
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, :-1] = arr[:, 1:] - arr[:, :-1]
    gy[:-1, :] = arr[1:, :] - arr[:-1, :]
    mag = np.sqrt(gx**2 + gy**2)
    return mag


def remove_background(image: Image.Image, use_rembg: bool = True) -> Image.Image:
    """
    移除图像背景。
    
    参数:
        image: 输入 PIL 图像。
        use_rembg: 是否使用 BackgroundRemover（如果可用）。
    
    返回:
        移除背景后的图像（RGBA 模式，透明背景）。
    """
    if use_rembg and HAS_BACKGROUND_REMOVER and BackgroundRemover is not None:
        # 如果图像是 RGB 模式，使用 BackgroundRemover
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)
            return image.convert("RGBA")
        # 如果已经是 RGBA，直接返回
        return image.convert("RGBA")
    else:
        # 如果没有 BackgroundRemover，返回原始图像（转换为 RGBA）
        return image.convert("RGBA")


def preprocess_image_for_model(image: Image.Image, config: Dict | None = None) -> Dict:
    """
    供流水线调用的高层图像预处理入口。

    参数:
        image: 输入 PIL 图像。
        config: 可选的预处理配置字典。
            - enable_background_removal: bool (默认 True) - 是否启用背景移除

    返回:
        字典，包含：
            - 'image': 预处理后的图像（RGB 或 RGBA）
            - 'edge_map': 简单的边缘图（numpy 数组）
    """
    # 检查是否启用背景移除
    enable_bg_removal = True
    if config:
        enable_bg_removal = config.get("enable_background_removal", True)
    
    # 背景移除（如果启用）
    if enable_bg_removal:
        image = remove_background(image, use_rembg=True)
    
    # 缩放和规范化（如果是 RGBA，先转换为 RGB 进行缩放，然后转换回 RGBA）
    if image.mode == 'RGBA':
        # 对于 RGBA 图像，保持透明度
        resized = ImageOps.fit(image, (512, 512), method=Image.BICUBIC)
    else:
        # 对于 RGB 图像，转换为 RGB 进行缩放
        resized = resize_and_normalize(image)
    
    # 计算边缘图（使用 RGB 版本）
    edges = compute_edge_map(resized.convert("RGB") if resized.mode == 'RGBA' else resized)
    
    return {"image": resized, "edge_map": edges}


__all__ = ["resize_and_normalize", "compute_edge_map", "preprocess_image_for_model", "remove_background"]


