"""
多视图图像到 3D 资产生成的命令行入口脚本。
支持 Hunyuan3D-2mv 等多视角控制模型。

示例:
    python scripts/run_mv_image2asset.py \
        --config configs/hunyuan3d_mv.yaml \
        --front path/to/front.png \
        --back path/to/back.png \
        --left path/to/left.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline.mv_pipeline import MultiViewPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行多视图图像到 3D 资产生成流程。")
    parser.add_argument("--config", type=str, default="configs/hunyuan3d_mv.yaml", help="YAML 配置文件路径。")
    
    # 视图参数
    parser.add_argument("--front", type=str, required=True, help="前视图图像路径。")
    parser.add_argument("--left", type=str, help="左视图图像路径。")
    parser.add_argument("--right", type=str, help="右视图图像路径。")
    parser.add_argument("--back", type=str, help="后视图图像路径。")
    
    parser.add_argument(
        "--format",
        type=str,
        default="glb",
        choices=["obj", "glb"],
        help="输出网格格式（默认: glb）。",
    )
    parser.add_argument("--name", type=str, default="asset_from_mv", help="输出文件的基础名称。")
    parser.add_argument("--no-rembg", action="store_true", help="禁用背景移除（默认启用）。")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 收集视图
    views = {}
    if args.front: views["front"] = args.front
    if args.left: views["left"] = args.left
    if args.right: views["right"] = args.right
    if args.back: views["back"] = args.back
    
    if not views:
        print("错误: 必须至少提供一个视图（推荐至少提供 --front）。")
        sys.exit(1)

    # 加载 Pipeline
    if not Path(args.config).exists():
        print(f"警告: 配置文件 {args.config} 不存在。将尝试使用默认设置，但模型路径可能不正确。")
        # 可以在这里创建一个临时的默认配置，指向 HuggingFace 的 Hunyuan3D-2mv
        # 但最好是让用户提供
    
    try:
        pipeline = MultiViewPipeline.from_config_file(args.config)
    except Exception as e:
        import traceback
        print(f"加载 Pipeline 失败: {e}")
        print("\n完整错误信息:")
        traceback.print_exc()
        print("\n请检查配置文件路径和内容。对于 Hunyuan3D-2mv，确保 model_path 设置正确。")
        sys.exit(1)

    print(f"开始处理，输入视图: {list(views.keys())}")
    
    try:
        result = pipeline.generate_from_multiview(
            views,
            output_name=args.name,
            file_format=args.format,
            do_remove_background=not args.no_rembg
        )
        
        print(f"\n生成成功！")
        print(f"Mesh saved to: {Path(result['mesh_path'])}")
        print(f"Metadata saved to: {Path(result['metadata_path'])}")
        if result["previews"]:
            print("Preview images:")
            for p in result["previews"]:
                print(f"  - {Path(p)}")
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n生成失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


