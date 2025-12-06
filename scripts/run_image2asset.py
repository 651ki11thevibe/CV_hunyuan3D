"""
图像到 3D 资产生成的命令行入口脚本。

示例:
    # 单张图像
    python scripts/run_image2asset.py --image path/to/image.png
    
    # 多视角图像（三个视角）
    python scripts/run_image2asset.py --front path/to/front.png --left path/to/left.png --back path/to/back.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

# 确保项目根目录在 sys.path 中，便于 `from core...` 形式的导入在任何工作目录下都生效。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core.io.inputs import load_single_image, load_multi_view_images
from core.pipeline.generation_pipeline import GenerationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行图像到 3D 资产生成流程。支持单张图像或多视角图像（front, left, back）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  单张图像:
    python scripts/run_image2asset.py --image path/to/image.png
  
  多视角图像（三个视角）:
    python scripts/run_image2asset.py --front path/to/front.png --left path/to/left.png --back path/to/back.png
        """
    )
    parser.add_argument("--config", type=str, default="configs/hunyuan3d_default.yaml", help="YAML 配置文件路径。")
    
    # 单张图像模式
    parser.add_argument("--image", type=str, help="输入图像路径（单张图像模式）。")
    
    # 多视角图像模式
    parser.add_argument("--front", type=str, help="前视图图像路径（多视角模式）。")
    parser.add_argument("--left", type=str, help="左视图图像路径（多视角模式）。")
    parser.add_argument("--back", type=str, help="后视图图像路径（多视角模式）。")
    
    parser.add_argument(
        "--format",
        type=str,
        default="glb",
        choices=["obj", "glb"],
        help="输出网格格式（默认: glb）。",
    )
    parser.add_argument("--name", type=str, default="asset_from_image", help="输出文件的基础名称。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 检查输入模式
    has_single = args.image is not None
    has_multi = any([args.front, args.left, args.back])
    
    if not has_single and not has_multi:
        parser = argparse.ArgumentParser()
        parser.error("必须提供 --image（单张图像）或 --front/--left/--back（多视角图像）之一。")
    
    if has_single and has_multi:
        parser = argparse.ArgumentParser()
        parser.error("不能同时指定 --image 和 --front/--left/--back。请选择单张图像模式或多视角模式。")
    
    # 如果是多视角模式，检查是否提供了所有三个视角
    if has_multi:
        if not all([args.front, args.left, args.back]):
            parser = argparse.ArgumentParser()
            parser.error("多视角模式需要同时提供 --front、--left 和 --back 三个视角。")
    
    pipeline = GenerationPipeline.from_config_file(args.config)
    
    # 加载输入
    if has_single:
        # 单张图像模式
        image = load_single_image(args.image)
    else:
        # 多视角模式
        image_paths = {
            "front": args.front,
            "left": args.left,
            "back": args.back
        }
        image = load_multi_view_images(image_paths, required_views=["front", "left", "back"])
    
    # 生成资产
    result = pipeline.generate_from_image(image, output_name=args.name, file_format=args.format)
    
    print(f"Mesh saved to: {Path(result['mesh_path'])}")
    print(f"Metadata saved to: {Path(result['metadata_path'])}")
    if result["previews"]:
        print("Preview images:")
        for p in result["previews"]:
            print(f"  - {Path(p)}")


if __name__ == "__main__":
    main()


