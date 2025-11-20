"""
3D 资产生成系统的输出工具模块。

主要功能：
- 将网格（可带纹理）导出为常见格式
- 生成轻量级预览渲染（当前为占位实现）

导出逻辑刻意保持简单，方便后续在不破坏对外 API 的前提下，
叠加高斯点云等其他表示形式的转换与导出。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image

from core.models.hunyuan3d_wrapper import RawAsset


def ensure_dir(path: str | Path) -> Path:
    """若目录不存在则创建，并返回对应的 `Path` 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_asset_as_mesh(
    asset: RawAsset,
    output_path: str | Path,
    *,
    file_format: str = "glb",
) -> Path:
    """
    将 `RawAsset` 保存为网格文件。

    参数:
        asset: 生成的资产对象，包含网格顶点与三角面。
        output_path: 目标文件路径（后缀可能会被 `file_format` 覆盖）。
        file_format: {"obj", "glb"} 之一。

    返回:
        实际写入的文件路径。
    """
    output_path = Path(output_path)
    if file_format not in {"obj", "glb"}:
        raise ValueError(f"Unsupported format: {file_format}。支持的格式: obj, glb")

    # 如果原始 mesh 可用且已有 visual 信息，直接使用（避免重新创建丢失信息）
    if hasattr(asset, "_original_mesh") and asset._original_mesh is not None:
        mesh = asset._original_mesh.copy()
        # 确保纹理文件已保存
        if asset.textures and asset.textures.get("albedo") is not None:
            _prepare_texture_image(asset.textures["albedo"], output_path)
    else:
        vertices = np.asarray(asset.mesh["vertices"], dtype=np.float32)
        faces = np.asarray(asset.mesh["faces"], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        _apply_textures_to_mesh(asset, mesh, output_path)

    # 设置正确的文件扩展名
    output_path = output_path.with_suffix(f".{file_format}")

    if file_format == "obj":
        mesh.export(output_path.as_posix())
    elif file_format == "glb":
        mesh.export(output_path.as_posix())

    return output_path


def save_asset_metadata(asset: RawAsset, output_dir: str | Path, name: str = "metadata") -> Path:
    """
    保存资产的轻量级元信息（JSON），便于调试或传递给后续模块。
    """
    import json

    def _to_serializable(obj: Any) -> Any:
        """
        递归地将 NumPy / 非 JSON 类型转换为可被 json 序列化的结构。
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, trimesh.Trimesh):
            # trimesh 对象无法序列化，返回基本信息
            return {"type": "Trimesh", "vertices": len(obj.vertices), "faces": len(obj.faces)}
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        return obj

    output_dir = ensure_dir(output_dir)
    meta_path = output_dir / f"{name}.json"
    raw_data = asdict(asset)
    # 移除 _original_mesh（trimesh 对象无法 JSON 序列化）
    raw_data.pop("_original_mesh", None)
    textures = raw_data.get("textures")
    original_textures = asset.textures or {}
    if textures:
        if "albedo" in textures:
            textures["albedo"] = {
                "shape": list(np.asarray(original_textures.get("albedo")).shape)
                if original_textures.get("albedo") is not None
                else None,
                "path": original_textures.get("albedo_path"),
            }
        if "uv" in textures:
            textures["uv"] = {
                "shape": list(np.asarray(original_textures.get("uv")).shape)
                if original_textures.get("uv") is not None
                else None,
            }
    data = _to_serializable(raw_data)
    meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return meta_path


def render_preview_images(asset: RawAsset, output_dir: str | Path, num_views: int = 4) -> list[Path]:
    """
    渲染资产的简单预览图像。

    TODO:
        使用离屏渲染器实现真实的多视图渲染
        （例如 pyrender、OpenGL 或可微渲染器）。

    目前为保持依赖轻量化，仅返回空列表，不做实际渲染。
    """
    _ = asset, output_dir, num_views  # keep signature stable
    return []


__all__ = [
    "ensure_dir",
    "save_asset_as_mesh",
    "save_asset_metadata",
    "render_preview_images",
]


def _apply_textures_to_mesh(asset: RawAsset, mesh: trimesh.Trimesh, output_path: Path) -> None:
    textures = asset.textures
    if not textures:
        return
    albedo = textures.get("albedo")
    uv = textures.get("uv")
    if albedo is None or uv is None:
        return
    image_array, texture_path = _prepare_texture_image(albedo, output_path)
    if texture_path is None:
        return
    textures["albedo_path"] = texture_path.as_posix()
    uv_array = np.asarray(uv, dtype=np.float32)
    material = trimesh.visual.texture.SimpleMaterial(image=image_array)
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv_array,
        image=image_array,
        material=material,
    )


def _prepare_texture_image(albedo: Any, output_path: Path) -> Tuple[np.ndarray, Optional[Path]]:
    array = np.asarray(albedo)
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    texture_path = output_path.with_name(f"{output_path.name}_albedo.png")
    Image.fromarray(array).save(texture_path)
    return array, texture_path


