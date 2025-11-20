"""
高层生成流水线模块。

整体流程:
    输入 -> 预处理 -> Hunyuan3D 推理
         -> （可选精炼）-> 导出

本模块负责在 IO、预处理、主干模型、精炼与导出之间进行调度，
同时保持各组件解耦，便于未来替换或扩展。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

import yaml
from PIL import Image

from core.io.outputs import ensure_dir, render_preview_images, save_asset_as_mesh, save_asset_metadata
from core.models.hunyuan3d_wrapper import RawAsset, load_hunyuan3d_from_config
from core.preprocess.image_preprocess import preprocess_image_for_model
from core.preprocess.text_preprocess import apply_prompt_engineering
from core.refine.mv_refine import refine as mv_refine


class GenerationPipeline:
    """
    负责将文本或图像输入转换为 3D 资产的总控流水线。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = load_hunyuan3d_from_config(config)
        self.output_dir = ensure_dir(config.get("default_output_dir", "outputs"))

    @classmethod
    def from_config_file(cls, path: str | Path) -> "GenerationPipeline":
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _run_refinement(self, asset: RawAsset) -> RawAsset:
        """
        调用可选的精炼步骤（当前仅为多视图占位实现）。
        """
        refined = mv_refine(asset, self.config)
        return refined

    def generate_from_text(
        self,
        prompt: str,
        *,
        output_name: str = "asset_from_text",
        file_format: Literal["obj", "glb"] = "glb",
    ) -> Dict[str, Any]:
        """
        完整的文本到资产生成流水线。

        返回:
            字典，包含:
                - 'asset': RawAsset 对象
                - 'mesh_path': 导出的网格文件路径
                - 'metadata_path': 元数据 JSON 路径
                - 'previews': 预览图路径列表
                - 'intermediate_image_path': 文生图阶段生成的 2D 图像路径（中间产物）
        """
        import torch
        
        # 在加载文生图模型前，将 Hunyuan3D 模型 offload 到 CPU 以释放显存
        if hasattr(self.model, 'shape_pipeline') and self.model.shape_pipeline is not None:
            if hasattr(self.model.shape_pipeline, 'to'):
                self.model.shape_pipeline.to("cpu")
            if hasattr(self.model, 'texture_pipeline') and self.model.texture_pipeline is not None:
                if hasattr(self.model.texture_pipeline, 'to'):
                    self.model.texture_pipeline.to("cpu")
            # 清理 GPU 缓存
            torch.cuda.empty_cache()
        
        processed_prompt = apply_prompt_engineering(prompt, config=self.config)
        generated_image = self._generate_image_from_text(processed_prompt)
        
        # 文生图完成后，清理文生图模型的显存
        torch.cuda.empty_cache()
        
        # 将 Hunyuan3D 模型移回 GPU
        if hasattr(self.model, 'shape_pipeline') and self.model.shape_pipeline is not None:
            if hasattr(self.model.shape_pipeline, 'to'):
                self.model.shape_pipeline.to(self.config.get("device", "cuda"))
            if hasattr(self.model, 'texture_pipeline') and self.model.texture_pipeline is not None:
                if hasattr(self.model.texture_pipeline, 'to'):
                    self.model.texture_pipeline.to(self.config.get("device", "cuda"))
        
        # 保存中间产物：文生图生成的 2D 图像
        intermediate_image_path = self._save_intermediate_image(generated_image, output_name)
        
        # 继续执行图像到 3D 流程
        result = self.generate_from_image(
            generated_image,
            output_name=output_name,
            file_format=file_format,
        )
        
        # 添加中间图像路径到返回结果
        result["intermediate_image_path"] = intermediate_image_path
        return result

    def generate_from_image(
        self,
        image,
        *,
        output_name: str = "asset_from_image",
        file_format: Literal["obj", "glb"] = "glb",
    ) -> Dict[str, Any]:
        """
        完整的图像到资产生成流水线。
        """
        # 从配置中获取图像预处理设置
        image_preprocessing_config = self.config.get("image_preprocessing", {})
        preprocessed = preprocess_image_for_model(image, config=image_preprocessing_config)
        raw_asset = self.model.generate_from_image(preprocessed)
        refined_asset = self._run_refinement(raw_asset)

        mesh_path = save_asset_as_mesh(
            refined_asset,
            self.output_dir / output_name,
            file_format=file_format,
        )
        meta_path = save_asset_metadata(refined_asset, self.output_dir, name=f"{output_name}_metadata")
        previews = render_preview_images(refined_asset, self.output_dir)

        return {
            "asset": refined_asset,
            "mesh_path": mesh_path,
            "metadata_path": meta_path,
            "previews": previews,
        }

    def _save_intermediate_image(self, image: Image.Image, output_name: str) -> Path:
        """
        保存文生图阶段生成的中间 2D 图像。

        参数:
            image: 生成的 PIL 图像
            output_name: 输出文件的基础名称

        返回:
            保存的图像文件路径
        """
        image_path = self.output_dir / f"{output_name}_intermediate.png"
        image.save(image_path, format="PNG")
        return image_path

    def _generate_image_from_text(self, prompt: str) -> Image.Image:
        """
        使用本地 diffusers 库进行文生图。
        """
        cfg = self.config.get("text_to_image") or {}
        if not cfg.get("enabled"):
            raise RuntimeError(
                "text_to_image.enabled = false，当前配置未打开文本到图像模块，"
                "因此无法执行文本到 3D。请在配置中启用该模块。"
            )
        provider = (cfg.get("provider") or "local_diffusers").lower()
        if provider != "local_diffusers":
            raise ValueError(
                f"当前仅支持 local_diffusers 提供者，配置中指定了: {provider}。"
                "请将 text_to_image.provider 设置为 'local_diffusers'。"
            )
        return self._text_to_image_local(prompt, cfg)

    def _text_to_image_local(self, prompt: str, cfg: Dict[str, Any]) -> Image.Image:
        """
        使用本地 diffusers 库进行文生图（适合服务器环境，模型会缓存到本地）。
        
        如果配置了 local_model_path，从指定路径加载模型（推荐，避免每次下载）。
        否则从 HuggingFace/魔塔社区自动下载模型权重（首次运行会下载，后续使用缓存）。
        
        推荐模型：black-forest-labs/FLUX.1-schnell（快速、质量好）
        """
        try:
            from diffusers import DiffusionPipeline
            import torch
        except ImportError:
            raise RuntimeError(
                "使用 local_diffusers 需要安装: pip install diffusers transformers accelerate"
            )

        # 优先使用本地路径，如果没有则使用 model_id（会从 HuggingFace/魔塔下载）
        local_model_path = cfg.get("local_model_path")
        model_id = cfg.get("model_id") or "black-forest-labs/FLUX.1-schnell"
        device = self.config.get("device", "cuda")
        dtype = torch.float16 if device == "cuda" else torch.float32
        low_vram = self.config.get("low_vram_mode", False)

        # 确定模型标识（用于缓存键）
        model_identifier = local_model_path if local_model_path else model_id
        cache_key = f"_local_pipeline_{model_identifier}"

        # 使用缓存避免重复加载（适合服务器长期运行）
        if not hasattr(self, cache_key):
            if local_model_path:
                print(f"[文生图] 从本地路径加载模型: {local_model_path}")
                from pathlib import Path
                if not Path(local_model_path).exists():
                    raise RuntimeError(
                        f"指定的本地模型路径不存在: {local_model_path}\n"
                        "请确保已从魔塔社区或其他来源下载模型到该路径"
                    )
            else:
                print(f"[文生图] 正在加载模型 {model_id}，首次运行会下载权重（仅一次）...")
            try:
                # 如果指定了本地路径，优先使用；否则从 HuggingFace/魔塔下载
                load_path = local_model_path if local_model_path else model_id
                
                # 本地路径加载时，不指定 variant（因为本地模型可能没有 fp16 变体）
                # 从 HuggingFace/魔塔下载时，也不强制 variant，让 diffusers 自动选择
                pipeline = DiffusionPipeline.from_pretrained(
                    load_path,
                    torch_dtype=dtype,
                )
                if device == "cuda" and torch.cuda.is_available():
                    if low_vram and hasattr(pipeline, "enable_model_cpu_offload"):
                        # 低显存模式：模型组件按需在 CPU/GPU 间移动
                        pipeline.enable_model_cpu_offload()
                        print("[文生图] 已启用低显存模式（CPU offload）")
                    else:
                        pipeline = pipeline.to(device)
                else:
                    pipeline = pipeline.to("cpu")
                setattr(self, cache_key, pipeline)
                print(f"[文生图] 模型加载完成，已缓存（后续运行无需重新加载）")
            except Exception as e:
                error_msg = str(e)
                # 检查是否是缺少依赖的问题
                if "sentencepiece" in error_msg.lower():
                    raise RuntimeError(
                        f"加载文生图模型失败: {error_msg}\n"
                        "缺少依赖 sentencepiece，请安装: pip install sentencepiece"
                    )
                elif "variant" in error_msg.lower() and "fp16" in error_msg.lower():
                    raise RuntimeError(
                        f"加载文生图模型失败: {error_msg}\n"
                        "本地模型可能没有 fp16 变体，请检查模型文件是否完整，"
                        "或尝试从 HuggingFace/魔塔重新下载完整模型。"
                    )
                else:
                    raise RuntimeError(
                        f"加载文生图模型失败: {error_msg}\n"
                        "请确保：1) 模型路径正确且文件完整 2) 有足够的磁盘空间 3) GPU 显存足够"
                    )
        pipeline = getattr(self, cache_key)

        # 构建生成参数
        parameters: Dict[str, Any] = {}
        for key in ("width", "height", "guidance_scale", "negative_prompt", "num_inference_steps"):
            value = cfg.get(key)
            if value not in (None, ""):
                parameters[key] = value

        # 生成图像
        try:
            result = pipeline(prompt, **parameters)
            if isinstance(result, (list, tuple)):
                image = result[0]
            else:
                image = getattr(result, "images", [None])[0]
            if image is None:
                raise RuntimeError("本地 diffusers 生成图像失败，返回 None。")
            # 生成完成后立即清理显存（如果使用 CPU offload，这一步会确保模型组件回到 CPU）
            import torch
            torch.cuda.empty_cache()
            return image.convert("RGB")
        except Exception as e:
            import torch
            torch.cuda.empty_cache()
            raise RuntimeError(f"文生图推理失败: {e}")


__all__ = ["GenerationPipeline"]


