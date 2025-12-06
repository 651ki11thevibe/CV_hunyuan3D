"""
模型下载脚本。
用于下载 Hunyuan3D-2mv 及其依赖的纹理模型到本地 `weights` 目录。
默认直接使用 Hugging Face 官方源，支持通过 --mirror 参数启用国内镜像。
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess

def install_huggingface_hub():
    try:
        import huggingface_hub
    except ImportError:
        print("正在安装 huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])

def download_model(repo_id, local_dir, subfolder=None, use_mirror=False):
    from huggingface_hub import snapshot_download
    
    print(f"\n正在下载模型: {repo_id} " + (f"(subfolder: {subfolder})" if subfolder else ""))
    print(f"目标路径: {local_dir}")
    
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("已启用 HF 镜像 (https://hf-mirror.com)")
    else:
        # 确保使用官方源
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        print("使用 Hugging Face 官方源")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
            resume_download=True,
            local_dir_use_symlinks=False  # 确保下载真实文件而不是软链接
        )
        print("下载完成！")
    except Exception as e:
        print(f"下载失败: {e}")
        if not use_mirror:
            print("如果遇到网络问题，可以尝试使用 --mirror 参数启用国内镜像。")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="下载 Hunyuan3D 模型权重")
    parser.add_argument("--target_dir", default="weights", help="模型保存目录 (默认: weights)")
    parser.add_argument("--mirror", action="store_true", help="启用 HF 国内镜像 (https://hf-mirror.com)")
    args = parser.parse_args()

    install_huggingface_hub()
    
    base_dir = Path(args.target_dir).resolve()
    base_dir.mkdir(exist_ok=True, parents=True)

    # 1. 下载 Hunyuan3D-2mv (Shape Model)
    # 只下载 hunyuan3d-dit-v2-mv 子文件夹
    # 注意：下载到 weights/tencent/Hunyuan3D-2mv 结构下，或者扁平化结构
    # 为了配合 diffusers 的加载逻辑，通常保持 repo 结构比较好，
    # 但 snapshot_download 的 local_dir 如果指定为 weights/Hunyuan3D-2mv，它会把内容直接放进去。
    
    # 我们希望最终目录结构是：
    # weights/
    #   hunyuan3d-dit-v2-mv/
    #   hunyuan3d-paint-v2-0/
    
    # 1. 下载 MV Shape 模型
    download_model(
        repo_id="tencent/Hunyuan3D-2mv",
        local_dir=str(base_dir),  # 将会下载到 weights/hunyuan3d-dit-v2-mv
        subfolder="hunyuan3d-dit-v2-mv",
        use_mirror=args.mirror
    )

    # 2. 下载 Texture 模型 (来自 Hunyuan3D-2)
    download_model(
        repo_id="tencent/Hunyuan3D-2",
        local_dir=str(base_dir),  # 将会下载到 weights/hunyuan3d-paint-v2-0
        subfolder="hunyuan3d-paint-v2-0",
        use_mirror=args.mirror
    )
    
    print("\n所有模型下载完成！")
    print(f"权重位于: {base_dir}")
    print("\n请使用以下配置运行 (推荐使用 configs/hunyuan3d_mv_local.yaml):")
    print(f"model_path: {base_dir}")
    print(f"model_subfolder: hunyuan3d-dit-v2-mv")
    print(f"texture_model_path: {base_dir}")
    print(f"texture_subfolder: hunyuan3d-paint-v2-0")

if __name__ == "__main__":
    main()

