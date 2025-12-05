from __future__ import annotations
from typing import Any, Dict, Literal, Union, List, Optional
from pathlib import Path
from PIL import Image
import torch
import os

from core.pipeline.mv_pipeline import MultiViewPipeline

# Try importing Qwen pipeline, handle case where it's missing
try:
    from diffusers import QwenImageEditPlusPipeline
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

class QwenEditMVPipeline(MultiViewPipeline):
    """
    集成 Qwen-Image-Edit 的生成流水线。
    先使用 Qwen 对输入图像进行编辑（如旋转、改色），再将编辑后的图像输入 Hunyuan3D 生成 3D 资产。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.edit_pipeline = None
        self.device = config.get("device", "cuda")
        
        if not QWEN_AVAILABLE:
            print("[QwenEditMVPipeline] Warning: 'diffusers' with QwenImageEditPlusPipeline support not found.")
            print("Please install: pip install git+https://github.com/huggingface/diffusers transformers>=4.51.3")

    def load_edit_model(self, model_id: str = "Qwen/Qwen-Image-Edit-2509"):
        """加载 Qwen 编辑模型"""
        if not QWEN_AVAILABLE:
            raise ImportError("QwenImageEditPlusPipeline not available.")

        if self.edit_pipeline is None:
            print(f"[QwenEditMVPipeline] Loading Qwen-Image-Edit model: {model_id}...")
            # 为了节省显存，使用 bfloat16
            self.edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16
            )
            self.edit_pipeline.to(self.device)
            self.edit_pipeline.set_progress_bar_config(disable=None)
            print("[QwenEditMVPipeline] Qwen Model loaded.")
    
    def unload_edit_model(self):
        """卸载编辑模型以释放显存"""
        if self.edit_pipeline is not None:
            del self.edit_pipeline
            torch.cuda.empty_cache()
            self.edit_pipeline = None
            print("[QwenEditMVPipeline] Qwen Model unloaded.")

    def edit_image(self, 
                   image: Image.Image, 
                   prompt: str, 
                   negative_prompt: str = " ",
                   num_inference_steps: int = 40,
                   guidance_scale: float = 1.0) -> Image.Image:
        """使用 Qwen 编辑图像"""
        if self.edit_pipeline is None:
            self.load_edit_model()
            
        # 确保图像是 RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = {
            "image": [image], 
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
            "generator": torch.Generator(device=self.device).manual_seed(42)
        }
        
        print(f"[QwenEditMVPipeline] Editing image with prompt: '{prompt}'...")
        with torch.inference_mode():
            output = self.edit_pipeline(**inputs)
            edited_image = output.images[0]
            
        return edited_image

    def edit_and_generate(
        self,
        image_path: Union[str, Path, Image.Image],
        edit_prompt: str,
        output_name: str = "asset_from_edited_qwen",
        file_format: Literal["obj", "glb"] = "glb",
        save_edited_image: bool = True
    ) -> Dict[str, Any]:
        """
        主流程: 编辑 -> 生成
        
        Args:
            image_path: 输入图像路径或 PIL 对象
            edit_prompt: 编辑提示词 (例如: "Rotate the object 90 degrees clockwise")
            output_name: 输出文件名
            file_format: 输出格式
        """
        # 1. 准备图像
        if isinstance(image_path, (str, Path)):
            original_img = Image.open(image_path).convert("RGB")
        else:
            original_img = image_path.convert("RGB")
            
        # 2. 执行编辑
        print(f"--- Step 1: Editing Image ---")
        edited_img = self.edit_image(original_img, prompt=edit_prompt)
        
        if save_edited_image:
            save_path = self.output_dir / f"{output_name}_edited.png"
            edited_img.save(save_path)
            print(f"[QwenEditMVPipeline] Saved edited image to {save_path}")

        # 3. 执行 MV 生成 (调用父类方法)
        print(f"--- Step 2: Generating 3D Asset from Edited Image ---")
        
        # 释放 Qwen 模型显存，避免 Hunyuan3D OOM
        self.unload_edit_model() 
        
        # 构造输入字典，此时我们把编辑后的图作为 'front' 视角
        # 也可以扩展逻辑，让 Qwen 生成多视角图，但这里先做单图输入
        mv_inputs = {
            "front": edited_img
        }
        
        result = self.generate_from_multiview(
            images=mv_inputs,
            output_name=output_name,
            file_format=file_format,
            do_remove_background=True # 再次移除背景，因为编辑后的图可能背景不干净
        )
        
        result["edited_image"] = edited_img
        return result
