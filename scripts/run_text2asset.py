"""
文本到 3D 资产生成的命令行入口脚本。

示例:
    python scripts/run_text2asset.py --prompt "a red chair"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

# 确保项目根目录在 sys.path 中，便于 `from core...` 形式的导入在任何工作目录下都生效。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.io.inputs import load_text_prompt
from core.pipeline.generation_pipeline import GenerationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行文本到 3D 资产生成流程。")
    parser.add_argument("--config", type=str, default="configs/hunyuan3d_default.yaml", help="YAML 配置文件路径。")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="直接输入的文本提示词。")
    group.add_argument("--prompt_file", type=str, help="包含提示词的文本文件路径。")
    parser.add_argument(
        "--format",
        type=str,
        default="glb",
        choices=["obj", "glb"],
        help="输出网格格式（默认: glb）。",
    )
    parser.add_argument("--name", type=str, default="asset_from_text", help="输出文件的基础名称。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = GenerationPipeline.from_config_file(args.config)
    prompt = load_text_prompt(prompt=args.prompt, prompt_file=args.prompt_file)

    result = pipeline.generate_from_text(prompt, output_name=args.name, file_format=args.format)
    
    # 输出中间产物（文生图生成的 2D 图像）
    if "intermediate_image_path" in result:
        print(f"Intermediate image (text-to-image) saved to: {Path(result['intermediate_image_path'])}")
    
    print(f"Mesh saved to: {Path(result['mesh_path'])}")
    print(f"Metadata saved to: {Path(result['metadata_path'])}")
    if result["previews"]:
        print("Preview images:")
        for p in result["previews"]:
            print(f"  - {Path(p)}")


if __name__ == "__main__":
    main()


