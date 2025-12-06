"""
基于 Hunyuan3D-2 主干的轻量封装模块。

本模块提供：
- `RawAsset`：网格 + 纹理的最小资产容器
- 便捷接口 `generate_from_text` 与 `generate_from_image`
- 向其他表示（如高斯点云）扩展的挂载点

当前版本直接调用官方 `hy3dgen` pipeline，只要在外部环境中安装好
Hunyuan3D-2 仓库即可复用真实推理能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import trimesh

try:  # Lazy import so that unit tests without torch still run.
    import torch
except ImportError:  # pragma: no cover - handled at runtime with clear error.
    torch = None  # type: ignore[assignment]

try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
except ImportError:  # pragma: no cover - handled via runtime error.
    Hunyuan3DDiTFlowMatchingPipeline = None  # type: ignore[assignment]

try:
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
except ImportError:  # pragma: no cover - optional dependency.
    Hunyuan3DPaintPipeline = None  # type: ignore[assignment]


@dataclass
class RawAsset:
    """
    用于承载生成 3D 资产的最小数据结构。

    属性:
        mesh: 包含以下必需字段的字典:
            - 'vertices': (N, 3) float32 顶点坐标
            - 'faces': (M, 3) int64 三角面索引
        textures: 可选纹理信息（如 UV 贴图、纹理图像等）。
        _original_mesh: 内部使用，保存原始 trimesh 对象（如果可用），用于直接导出。
    """

    mesh: Dict[str, Any]
    textures: Optional[Dict[str, Any]] = field(default=None)
    _original_mesh: Any = field(default=None, repr=False)


def _resolve_torch_dtype(dtype: Optional[str]):
    if torch is None or dtype is None:
        return None
    normalized = dtype.lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(normalized)


def _unwrap_pipeline_output(result: Any) -> Any:
    """
    Hunyuan3D 的 diffusers 风格 pipeline 可能返回 list/tuple、dataclass
    或直接返回 Trimesh。该函数统一为单个 mesh-like 对象。
    """
    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("Pipeline 返回空结果，无法构建 RawAsset。")
        return result[0]
    for attr in ("meshes", "mesh", "geometry"):
        if hasattr(result, attr):
            candidate = getattr(result, attr)
            if isinstance(candidate, (list, tuple)):
                if not candidate:
                    continue
                return candidate[0]
            return candidate
    return result


def _select_geometry_from_scene(scene: trimesh.Scene) -> trimesh.Trimesh:
    """
    从场景中选择主要几何体，过滤掉辅助平面。
    """
    geometries = []
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        geometries.append((name or "", geom))
    if not geometries:
        return scene.dump(concatenate=True)
    # 过滤 plane
    filtered = [
        (name, geom)
        for name, geom in geometries
        if "plane" not in name.lower()
    ]
    target_list = filtered if filtered else geometries
    name, geom = max(target_list, key=lambda item: int(item[1].faces.shape[0]))
    return geom


def _ensure_trimesh(mesh_like: Any) -> trimesh.Trimesh:
    if isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like
    if isinstance(mesh_like, trimesh.Scene):
        return _select_geometry_from_scene(mesh_like)
    raise TypeError(f"无法将类型 {type(mesh_like)!r} 解析为 trimesh.Trimesh。")


def _extract_texture_data(mesh: trimesh.Trimesh) -> Optional[Dict[str, Any]]:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    texture_payload: Dict[str, Any] = {}
    if hasattr(visual, "uv") and visual.uv is not None:
        texture_payload["uv"] = np.asarray(visual.uv, dtype=np.float32)
    material = getattr(visual, "material", None)
    if material is not None:
        image = getattr(material, "image", None)
        if image is not None:
            texture_payload["albedo"] = np.asarray(image)
        image_path = getattr(material, "image_path", None)
        if image_path:
            texture_payload["image_path"] = image_path
    return texture_payload or None


class Hunyuan3DModel:
    """
    Hunyuan3D-2 主干模型的薄封装类，直接复用官方 `hy3dgen` pipeline。
    """

    def __init__(
        self,
        model_path: str,
        *,
        model_subfolder: Optional[str] = None,
        device: str = "cuda",
        sampling_steps: int = 30,
        torch_dtype: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
        enable_texture: bool = False,
        texture_model_path: Optional[str] = None,
        texture_subfolder: Optional[str] = None,
        texture_reference_field: str = "image",
        low_vram_mode: bool = False,
        enable_flashvdm: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("未检测到 PyTorch，请先按照官方说明安装 GPU 版本的 PyTorch。")
        if Hunyuan3DDiTFlowMatchingPipeline is None:
            raise RuntimeError(
                "未找到 hy3dgen 模块。请进入官方 Hunyuan3D-2 仓库执行 `pip install -e .` 后再运行本项目。"
            )

        self.device = device
        self.sampling_steps = sampling_steps
        self.texture_reference_field = texture_reference_field
        self.low_vram_mode = low_vram_mode
        self.enable_flashvdm = enable_flashvdm
        self._hf_auth_token = hf_auth_token
        self._torch_dtype = _resolve_torch_dtype(torch_dtype)

        variant = None
        self.shape_pipeline = self._load_shape_pipeline(model_path, model_subfolder, variant=variant)
        self.texture_pipeline = None

        if enable_texture and texture_model_path:
            if Hunyuan3DPaintPipeline is None:
                raise RuntimeError(
                    "配置中启用了纹理模型，但当前环境未安装 `hy3dgen.texgen`。"
                    " 请在 Hunyuan3D-2 仓库中按照 README 安装 texgen 相关依赖。"
                )
            self.texture_pipeline = self._load_texture_pipeline(texture_model_path, texture_subfolder)
    
    def _load_shape_pipeline(self, model_path: str, model_subfolder: Optional[str], variant: Optional[str] = None):
        kwargs: Dict[str, Any] = {}
        if model_subfolder:
            kwargs["subfolder"] = model_subfolder
        if variant:
            kwargs["variant"] = variant
        if self._hf_auth_token:
            kwargs["token"] = self._hf_auth_token

        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        pipe.to(self.device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._configure_optional_modes(pipe)
        return pipe

    def _load_texture_pipeline(self, model_path: str, model_subfolder: Optional[str]):
        kwargs: Dict[str, Any] = {}
        if model_subfolder:
            kwargs["subfolder"] = model_subfolder
        if self._hf_auth_token:
            kwargs["token"] = self._hf_auth_token

        pipe = Hunyuan3DPaintPipeline.from_pretrained(model_path, **kwargs)
        if hasattr(pipe, "to"):
            pipe.to(self.device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._configure_optional_modes(pipe)
        return pipe

    def _configure_optional_modes(self, pipe: Any) -> None:
        if self.low_vram_mode and hasattr(pipe, "enable_low_vram_mode"):
            pipe.enable_low_vram_mode()
        if self.enable_flashvdm and hasattr(pipe, "enable_flashvdm"):
            pipe.enable_flashvdm()

    def _maybe_apply_texture(self, mesh: Any, reference: Any) -> Any:
        """
        应用纹理到网格，与官方 demo 保持一致：
        mesh = pipeline_texgen(mesh, image=image)
        """
        if self.texture_pipeline is None or reference is None:
            return mesh
        mesh_for_texture = _ensure_trimesh(_unwrap_pipeline_output(mesh))
        outputs = self.texture_pipeline(mesh_for_texture, image=reference)
        return _unwrap_pipeline_output(outputs)

    def _mesh_to_raw_asset(self, mesh_like: Any) -> RawAsset:
        """
        将 trimesh 对象转换为 RawAsset。
        
        注意：如果原始 mesh 已经有 visual 信息，我们会保留它以便后续直接导出。
        """
        trimesh_obj = _ensure_trimesh(_unwrap_pipeline_output(mesh_like))
        vertices = np.asarray(trimesh_obj.vertices, dtype=np.float32)
        faces = np.asarray(trimesh_obj.faces, dtype=np.int64)
        textures = _extract_texture_data(trimesh_obj)
        
        # 如果原始 mesh 有 visual 信息，保存原始 mesh 引用以便直接导出
        raw_asset = RawAsset(mesh={"vertices": vertices, "faces": faces}, textures=textures)
        if hasattr(trimesh_obj, "visual") and trimesh_obj.visual is not None:
            # 保存原始 mesh 引用，以便在导出时直接使用（避免重新创建丢失信息）
            raw_asset._original_mesh = trimesh_obj
        return raw_asset

    def _resolve_texture_reference(
        self,
        extra_cond: Optional[Dict[str, Any]],
        fallback: Any = None,
    ) -> Any:
        if not self.texture_pipeline:
            return None
        if extra_cond:
            if "texture_reference" in extra_cond:
                return extra_cond["texture_reference"]
            if self.texture_reference_field in extra_cond:
                return extra_cond[self.texture_reference_field]
        return fallback

    def generate_from_text(self, prompt: str, extra_cond: Optional[Dict[str, Any]] = None) -> RawAsset:
        """
        Hunyuan3D-2 v2.0 主要支持 Image-to-3D，不支持直接文本输入。
        请使用 generate_from_image 或通过文本到图像模型生成图像后再调用。
        """
        raise RuntimeError(
            "当前配置的 Hunyuan3D-2 v2.0 形状模型主要支持『图像到 3D』，"
            "原生 pipeline 仅依赖图像编码器。要使用本项目生成 3D，"
            "请改用 `scripts/run_image2asset.py` 并提供输入图像；"
            "若确需『文本到 3D』，需要额外接入文本→图像模型或 "
            "Hunyuan3D 1.x 的文本管线。"
        )

    def generate_from_image(
        self,
        preprocessed_image: Dict[str, Any],
        extra_cond: Optional[Dict[str, Any]] = None,
    ) -> RawAsset:
        """
        图像到 3D 推理。
        
        参数:
            preprocessed_image: 包含 "image" 字段的字典，值为 PIL.Image 或视图字典
            extra_cond: 额外参数，支持 output_type, octree_resolution, num_chunks, generator
        """
        image = preprocessed_image.get("image")
        if image is None:
            raise ValueError("预处理结构缺少 `image` 字段，无法执行图像到 3D 推理。")
        
        # 构建 shape pipeline 的调用参数
        shape_kwargs = {
            "image": image,
            "num_inference_steps": self.sampling_steps,
        }
        
        shape_pipeline_kwargs = getattr(self, "shape_pipeline_kwargs", {})
        if shape_pipeline_kwargs:
            shape_kwargs.update(shape_pipeline_kwargs)
        if extra_cond:
            if "output_type" in extra_cond:
                shape_kwargs["output_type"] = extra_cond["output_type"]
            for key in ["octree_resolution", "num_chunks", "generator"]:
                if key in extra_cond:
                    shape_kwargs[key] = extra_cond[key]
        
        shape_result = self.shape_pipeline(**shape_kwargs)
        shape = _unwrap_pipeline_output(shape_result)
        
        fallback_reference = preprocessed_image.get(self.texture_reference_field)
        
        # 如果 fallback_reference 是多视角图像字典，使用 "front" 视角作为纹理参考（与官方 demo 一致）
        if isinstance(fallback_reference, dict) and "front" in fallback_reference:
            fallback_reference = fallback_reference["front"]
        
        reference = self._resolve_texture_reference(extra_cond, fallback=fallback_reference)
        textured = self._maybe_apply_texture(shape, reference)
        return self._mesh_to_raw_asset(textured)


def load_hunyuan3d_from_config(config: Dict[str, Any]) -> Hunyuan3DModel:
    """
    从配置字典构建 Hunyuan3D 模型实例的辅助函数。
    """
    model = Hunyuan3DModel(
        model_path=config.get("model_path", "path/to/hunyuan3d"),
        model_subfolder=config.get("model_subfolder"),
        device=config.get("device", "cuda"),
        sampling_steps=int(config.get("sampling_steps", 30)),
        torch_dtype=config.get("torch_dtype"),
        hf_auth_token=config.get("hf_auth_token"),
        enable_texture=bool(config.get("enable_texture", False)),
        texture_model_path=config.get("texture_model_path") or config.get("model_path"),
        texture_subfolder=config.get("texture_subfolder"),
        texture_reference_field=config.get("texture_reference_field", "image"),
        low_vram_mode=bool(config.get("low_vram_mode", False)),
        enable_flashvdm=bool(config.get("enable_flashvdm", False)),
    )
    model.shape_pipeline_kwargs = config.get("shape_pipeline_kwargs", {})
    if config.get("variant"):
        model.shape_pipeline = model._load_shape_pipeline(
            config.get("model_path", "path/to/hunyuan3d"),
            config.get("model_subfolder"),
            variant=config.get("variant")
        )
    return model


def convert_mesh_to_gaussians(asset: RawAsset, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    将网格形式的 `RawAsset` 转换为高斯点云表示的占位函数。

    参数:
        asset: 输入的网格资产。
        config: 可选的转换配置。

    返回:
        表示高斯参数的占位字典。

    TODO:
        实现真正的网格到 3D 高斯点云的转换，输出位置、尺度、
        透明度等参数，以适配高斯渲染相关的研究。
    """
    _ = asset, config
    return {
        "positions": np.zeros((1, 3), dtype=np.float32),
        "scales": np.ones((1, 3), dtype=np.float32),
        "opacities": np.ones((1, 1), dtype=np.float32),
    }


__all__ = [
    "RawAsset",
    "Hunyuan3DModel",
    "load_hunyuan3d_from_config",
    "convert_mesh_to_gaussians",
]


