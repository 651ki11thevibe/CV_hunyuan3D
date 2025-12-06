"""
使用Blender进行3D模型后处理的模块。

功能包括：
- 移除重复顶点
- 删除孤立顶点和边
- 删除退化面
- 修复法线方向
- 修复非流形边
- 自动填充孔洞
- 初步平滑（移除凸起）
- 简化模型
- 最终表面平滑
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from core.io.outputs import ensure_dir


def postprocess_with_blender(
    input_mesh_path: str | Path,
    output_mesh_path: str | Path,
    *,
    blender_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    使用Blender对3D模型进行后处理。

    参数:
        input_mesh_path: 输入的网格文件路径（.glb或.obj）
        output_mesh_path: 输出的网格文件路径
        blender_path: Blender可执行文件路径（如果为None，会尝试自动查找）
        config: 后处理配置字典

    返回:
        处理后文件的路径
    """
    input_mesh_path = Path(input_mesh_path)
    output_mesh_path = Path(output_mesh_path)
    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取Blender脚本路径
    script_path = Path(__file__).parent / "blender_postprocess_script.py"

    # 确定Blender路径
    if blender_path is None:
        blender_path = _find_blender_executable()
    if blender_path is None:
        raise RuntimeError(
            "未找到Blender可执行文件。请安装Blender或通过blender_path参数指定路径。\n"
            "安装方法：\n"
            "  - macOS: brew install --cask blender\n"
            "  - Linux: sudo apt-get install blender 或从官网下载\n"
            "  - Windows: 从 https://www.blender.org/download/ 下载安装"
        )

    # 准备配置
    if config is None:
        config = {}
    
    # 构建Blender命令
    # 使用--background模式，不打开GUI
    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(script_path),
        "--",
        str(input_mesh_path),
        str(output_mesh_path),
        json.dumps(config),
    ]

    # 执行Blender脚本
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            print("[Blender后处理] 输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        error_msg = f"Blender后处理失败: {e}\n"
        if e.stderr:
            error_msg += f"错误信息: {e.stderr}\n"
        if e.stdout:
            error_msg += f"输出: {e.stdout}\n"
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        raise RuntimeError(
            f"未找到Blender可执行文件: {blender_path}\n"
            "请确保Blender已正确安装并在PATH中，或通过blender_path参数指定完整路径。"
        )

    if not output_mesh_path.exists():
        raise RuntimeError(f"后处理完成，但输出文件不存在: {output_mesh_path}")

    return output_mesh_path


def _find_blender_executable() -> Optional[str]:
    """
    尝试自动查找Blender可执行文件。

    返回:
        Blender可执行文件路径，如果未找到则返回None
    """
    import shutil

    # 常见名称
    possible_names = ["blender", "blender.exe"]
    
    for name in possible_names:
        path = shutil.which(name)
        if path:
            return path
    
    # 尝试常见安装路径
    import platform
    
    system = platform.system()
    common_paths = []
    
    if system == "Darwin":  # macOS
        common_paths = [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender.app/Contents/MacOS/blender",
        ]
    elif system == "Linux":
        common_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/opt/blender/blender",
        ]
    elif system == "Windows":
        common_paths = [
            "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",
            "C:\\Program Files (x86)\\Blender Foundation\\Blender\\blender.exe",
        ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    return None


__all__ = ["postprocess_with_blender"]

