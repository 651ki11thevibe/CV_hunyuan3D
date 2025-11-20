"""
图像到 3D 资产生成的命令行入口脚本。

示例:
    python scripts/run_image2asset.py --image path/to/image.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中，便于 `from core...` 形式的导入在任何工作目录下都生效。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core.io.inputs import load_single_image
from core.pipeline.generation_pipeline import GenerationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行图像到 3D 资产生成流程。")
    parser.add_argument("--config", type=str, default="configs/hunyuan3d_default.yaml", help="YAML 配置文件路径。")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径。")
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
    pipeline = GenerationPipeline.from_config_file(args.config)
    image = load_single_image(args.image)

    result = pipeline.generate_from_image(image, output_name=args.name, file_format=args.format)
    print(f"Mesh saved to: {Path(result['mesh_path'])}")
    print(f"Metadata saved to: {Path(result['metadata_path'])}")
    if result["previews"]:
        print("Preview images:")
        for p in result["previews"]:
            print(f"  - {Path(p)}")


if __name__ == "__main__":
    main()


