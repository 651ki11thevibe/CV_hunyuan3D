"""
GRN推理脚本
用于对粗糙3D模型进行几何精炼
"""

import os
import torch
import numpy as np
import trimesh
import argparse
from PIL import Image

from models import create_grn_model
from data import MeshRenderer, GeometryDegradation
from utils.checkpoint import load_checkpoint
from utils.visualization import visualize_results


def load_coarse_mesh(mesh_path: str) -> trimesh.Trimesh:
    """加载粗糙网格"""
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def mesh_to_input_tensor(
    mesh: trimesh.Trimesh,
    renderer: MeshRenderer,
    camera_pose: np.ndarray
) -> torch.Tensor:
    """将网格渲染为输入张量"""
    depth, normal, mask = renderer.render_mesh(mesh, camera_pose)
    
    # 组合为5通道输入
    input_channels = np.concatenate([
        depth[..., np.newaxis],
        normal,
        mask[..., np.newaxis]
    ], axis=-1)
    
    input_tensor = torch.from_numpy(input_channels).float().permute(2, 0, 1).unsqueeze(0)
    return input_tensor


def refine_geometry(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """使用GRN精炼几何"""
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    return output


def output_to_depth_normal(
    output: torch.Tensor
) -> tuple:
    """从输出张量提取深度和法线"""
    output_np = output.squeeze(0).cpu().numpy()
    depth = output_np[0]
    normal = output_np[1:4].transpose(1, 2, 0)
    return depth, normal


def main():
    parser = argparse.ArgumentParser(description='GRN推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入粗糙网格路径')
    parser.add_argument('--output', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--image_size', type=int, default=256,
                       help='渲染图像尺寸')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='是否保存可视化结果')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_grn_model()
    model = model.to(device)
    
    # 加载检查点
    load_checkpoint(args.checkpoint, model, device=device)
    
    # 加载输入网格
    print(f"加载网格: {args.input}")
    mesh = load_coarse_mesh(args.input)
    
    # 创建渲染器
    renderer = MeshRenderer(image_size=args.image_size)
    
    # 生成相机位姿（可以从多个视角渲染）
    # 这里使用一个固定视角作为示例
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 2]  # 相机位置
    
    # 渲染输入
    print("渲染输入几何...")
    input_tensor = mesh_to_input_tensor(mesh, renderer, camera_pose)
    
    # 精炼几何
    print("精炼几何...")
    output = refine_geometry(model, input_tensor, device)
    
    # 提取结果
    depth_refined, normal_refined = output_to_depth_normal(output)
    
    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    
    # 保存深度图
    depth_img = Image.fromarray((depth_refined * 255).astype(np.uint8))
    depth_img.save(os.path.join(args.output, 'depth_refined.png'))
    
    # 保存法线图
    normal_img = Image.fromarray(((normal_refined + 1) / 2 * 255).astype(np.uint8))
    normal_img.save(os.path.join(args.output, 'normal_refined.png'))
    
    print(f"结果已保存到: {args.output}")
    
    # 可视化
    if args.visualize:
        # 获取原始输入用于对比
        depth_original, normal_original, mask_original = renderer.render_mesh(mesh, camera_pose)
        
        input_full = input_tensor.squeeze(0)
        target_full = torch.from_numpy(
            np.concatenate([
                depth_original[..., np.newaxis],
                normal_original
            ], axis=-1)
        ).float().permute(2, 0, 1)
        mask_full = torch.from_numpy(mask_original).float()
        
        visualize_results(
            input_full,
            output.squeeze(0),
            target_full,
            mask_full,
            save_path=os.path.join(args.output, 'visualization.png'),
            show=False
        )
        print("可视化结果已保存")


if __name__ == "__main__":
    main()

