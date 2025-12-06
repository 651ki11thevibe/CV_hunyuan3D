"""
3D 资产生成系统的输入工具模块。

本模块保持极简，只负责：
- 文本提示词读取
- 单张图像加载
- 多视角图像加载

设计上尽量轻量，以便后续轻松扩展更复杂的输入形式
（例如多视角图像、相机位姿等），而无需修改对外接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from PIL import Image


def load_text_prompt(prompt: Optional[str] = None, *, prompt_file: Optional[str] = None) -> str:
    """
    从字符串或文件中读取文本提示词。

    参数:
        prompt: 用户直接传入的提示词字符串。
        prompt_file: 包含提示词的文本文件路径。

    返回:
        处理后的提示词字符串。
    """
    if prompt is not None:
        return prompt.strip()

    if prompt_file is None:
        raise ValueError("Either `prompt` or `prompt_file` must be provided.")

    text = Path(prompt_file).read_text(encoding="utf-8")
    return text.strip()


def load_single_image(image_path: str) -> Image.Image:
    """
    从给定路径加载单张 RGB 图像。

    参数:
        image_path: 图像文件路径。

    返回:
        RGB 模式的 PIL Image 对象。
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path).convert("RGB")
    return img


def load_multi_view_images(
    image_paths: Dict[str, str],
    *,
    required_views: Optional[list[str]] = None,
) -> Dict[str, Image.Image]:
    """
    从多个路径加载多视角图像，用于多视角 3D 生成。

    参数:
        image_paths: 字典，键为视角名称（如 "front", "left", "back"），值为图像路径。
        required_views: 可选的必需视角列表。如果提供，将检查所有必需视角是否都存在。

    返回:
        字典，键为视角名称，值为 PIL Image 对象（RGBA 模式，用于背景移除）。

    示例:
        images = load_multi_view_images({
            "front": "assets/front.png",
            "left": "assets/left.png",
            "back": "assets/back.png"
        })
    """
    if required_views is None:
        required_views = []

    # 检查必需视角
    for view in required_views:
        if view not in image_paths:
            raise ValueError(f"Required view '{view}' is missing in image_paths.")

    # 加载所有视角的图像
    images: Dict[str, Image.Image] = {}
    for view_name, image_path in image_paths.items():
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found for view '{view_name}': {image_path}")

        # 加载为 RGBA 模式（用于后续背景移除）
        img = Image.open(path).convert("RGBA")
        images[view_name] = img

    return images


__all__ = ["load_text_prompt", "load_single_image", "load_multi_view_images"]


