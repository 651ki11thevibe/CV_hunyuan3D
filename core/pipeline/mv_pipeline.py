"""
Hunyuan3D-2mv 多视图生成流水线模块。

本模块扩展了通用的生成流水线，增加了对多视图（Multi-view）输入的支持。
适用于 Hunyuan3D-2mv 等模型，允许用户提供前视图、左视图、后视图等信息
以精确控制生成结果。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Union

from PIL import Image

from core.io.outputs import save_asset_as_mesh, save_asset_metadata, render_preview_images
from core.pipeline.generation_pipeline import GenerationPipeline
from core.preprocess.image_preprocess import remove_background, resize_and_normalize


class MultiViewPipeline(GenerationPipeline):
    """
    支持多视图输入的 Hunyuan3D 生成流水线。
    
    用法示例:
        pipeline = MultiViewPipeline(config)
        pipeline.generate_from_multiview({
            "front": "path/to/front.png",
            "left": "path/to/left.png",
            "back": "path/to/back.png"
        })
    """

    def generate_from_multiview(
        self,
        images: Dict[str, Union[str, Path, Image.Image]],
        *,
        output_name: str = "asset_from_mv",
        file_format: Literal["obj", "glb"] = "glb",
        do_remove_background: bool = True,
    ) -> Dict[str, Any]:
        """
        执行多视图到 3D 资产的生成。

        参数:
            images: 包含视图名称和对应图像（路径或 PIL 对象）的字典。
                   键通常为 "front", "left", "right", "back" 等。
            output_name: 输出文件的基础名称。
            file_format: 导出网格的格式 ("obj" 或 "glb")。
            do_remove_background: 是否对每张输入图像执行背景移除。

        返回:
            包含生成资产、路径和预览图的字典。
        """
        # 1. 预处理所有视图
        processed_views: Dict[str, Image.Image] = {}
        
        image_preprocessing_config = self.config.get("image_preprocessing", {})
        # 优先使用函数参数控制，其次查看配置
        should_rembg = do_remove_background and image_preprocessing_config.get("enable_background_removal", True)

        for view_name, img_input in images.items():
            # 加载图像
            if isinstance(img_input, (str, Path)):
                img = Image.open(img_input)
            elif isinstance(img_input, Image.Image):
                img = img_input
            else:
                raise TypeError(f"视图 '{view_name}' 的输入类型不支持: {type(img_input)}")

            # 移除背景
            if should_rembg:
                img = remove_background(img, use_rembg=True)
            
            # 统一尺寸和格式 (RGB/RGBA)
            # Hunyuan3D-2mv 内部处理逻辑通常需要 RGB，但保留 Alpha 通道有助于背景处理
            # 这里我们确保它被 resize 到 512x512 (官方 demo 常用尺寸，也可由模型内部处理)
            # 为保险起见，我们进行基本的 resize，但保持 RGBA 模式如果存在
            if img.mode == 'RGBA':
                from PIL import ImageOps
                img = ImageOps.fit(img, (512, 512), method=Image.BICUBIC)
                # 注意：很多 MV 模型期望白色背景的 RGB 图像，或者带 Mask 的输入
                # Hunyuan3DDiTFlowMatchingPipeline 内部通常会处理
                # 如果模型需要 RGB，我们可以在这里转换，给个白色背景
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = resize_and_normalize(img)

            processed_views[view_name] = img

        # 2. 调用模型进行生成
        # 注意：我们将字典传递给 'image' 字段，Hunyuan3DModel 需要透传这个字典
        # 给底层的 Hunyuan3DDiTFlowMatchingPipeline
        
        print(f"[MultiViewPipeline] 正在使用 {len(processed_views)} 个视图进行生成: {list(processed_views.keys())}...")
        
        # 构造 preprocessed_image 结构
        # 我们这里不需要 edge_map，因为 MV 模式下模型自己处理特征
        # 关键修改：纹理生成模型接受 List[Image] 或 Image，但不接受 Dict
        # 根据用户需求，我们将所有视图作为列表传递，以支持多图输入，同时也兼容单图
        texture_ref = list(processed_views.values()) if processed_views else None

        preprocessed_input = {
            "image": processed_views
        }

        # 通过 extra_cond 显式传递 texture_reference
        extra_cond = {
            "texture_reference": texture_ref
        }

        print(f"[MultiViewPipeline] 纹理参考图数量: {len(texture_ref) if texture_ref else 0}")
        
        raw_asset = self.model.generate_from_image(preprocessed_input, extra_cond=extra_cond)
        
        # 3. 可选精炼
        refined_asset = self._run_refinement(raw_asset)

        # 4. 导出与保存
        mesh_path = save_asset_as_mesh(
            refined_asset,
            self.output_dir / output_name,
            file_format=file_format,
        )
        
        # 保存元数据
        meta_path = save_asset_metadata(
            refined_asset, 
            self.output_dir, 
            name=f"{output_name}_metadata",
            extra_info={"views": list(processed_views.keys())}
        )
        
        # 渲染预览
        previews = render_preview_images(refined_asset, self.output_dir)

        return {
            "asset": refined_asset,
            "mesh_path": mesh_path,
            "metadata_path": meta_path,
            "previews": previews,
        }

