"""
准备真实训练数据
支持从Objaverse下载或使用本地数据
"""

import os
import shutil
import argparse
from pathlib import Path
import random

def download_objaverse_data(data_root: str, num_samples: int = 500):
    """从Objaverse下载数据"""
    try:
        import objaverse
        from tqdm import tqdm
        
        print("=" * 60)
        print("从Objaverse下载数据")
        print("=" * 60)
        
        # 创建目录
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        print(f"正在获取Objaverse对象列表...")
        # 获取Objaverse对象列表
        objects = objaverse.load_objects()
        
        # 随机选择样本
        selected_uids = random.sample(list(objects.keys()), min(num_samples, len(objects)))
        
        print(f"正在下载 {len(selected_uids)} 个对象...")
        downloaded = 0
        
        for uid in tqdm(selected_uids[:num_samples]):
            try:
                # 下载对象
                obj_path = objaverse.load_objects([uid])[uid]
                
                # 复制到训练或验证集（80/20分割）
                if random.random() < 0.8:
                    dest = os.path.join(train_dir, f"{uid}.obj")
                else:
                    dest = os.path.join(val_dir, f"{uid}.obj")
                
                # 如果下载的是目录，找其中的.obj文件
                if os.path.isdir(obj_path):
                    obj_files = list(Path(obj_path).rglob("*.obj"))
                    if obj_files:
                        shutil.copy(str(obj_files[0]), dest)
                        downloaded += 1
                elif obj_path.endswith('.obj'):
                    shutil.copy(obj_path, dest)
                    downloaded += 1
                    
            except Exception as e:
                print(f"下载 {uid} 失败: {e}")
                continue
        
        print(f"\n✓ 成功下载 {downloaded} 个对象")
        print(f"训练集: {len(os.listdir(train_dir))} 个文件")
        print(f"验证集: {len(os.listdir(val_dir))} 个文件")
        
    except ImportError:
        print("objaverse未安装")
        print("安装方法: pip install objaverse")
        return False
    except Exception as e:
        print(f"下载失败: {e}")
        return False
    
    return True


def prepare_from_directory(data_root: str, source_dir: str, train_ratio: float = 0.8):
    """从本地目录准备数据"""
    print("=" * 60)
    print("从本地目录准备数据")
    print("=" * 60)
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"错误: 源目录不存在: {source_dir}")
        return False
    
    # 创建目标目录
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 查找所有.obj文件
    obj_files = list(source_path.rglob("*.obj"))
    obj_files += list(source_path.rglob("*.ply"))
    obj_files += list(source_path.rglob("*.stl"))
    
    if not obj_files:
        print(f"错误: 在 {source_dir} 中未找到3D模型文件")
        return False
    
    print(f"找到 {len(obj_files)} 个3D模型文件")
    
    # 随机打乱
    random.shuffle(obj_files)
    
    # 分割训练和验证集
    split_idx = int(len(obj_files) * train_ratio)
    train_files = obj_files[:split_idx]
    val_files = obj_files[split_idx:]
    
    # 复制文件
    print("复制训练集文件...")
    for i, src_file in enumerate(train_files):
        dest = os.path.join(train_dir, f"mesh_{i:05d}{src_file.suffix}")
        try:
            shutil.copy(str(src_file), dest)
        except Exception as e:
            print(f"复制失败 {src_file}: {e}")
    
    print("复制验证集文件...")
    for i, src_file in enumerate(val_files):
        dest = os.path.join(val_dir, f"mesh_{i:05d}{src_file.suffix}")
        try:
            shutil.copy(str(src_file), dest)
        except Exception as e:
            print(f"复制失败 {src_file}: {e}")
    
    print(f"\n✓ 数据准备完成")
    print(f"训练集: {len(os.listdir(train_dir))} 个文件")
    print(f"验证集: {len(os.listdir(val_dir))} 个文件")
    
    return True


def download_sample_data(data_root: str, num_samples: int = 1000):
    """下载示例数据（使用公开数据集）"""
    print("=" * 60)
    print("准备示例数据")
    print("=" * 60)
    print("注意: 这是一个简化版本，实际使用时建议使用Objaverse或ShapeNet")
    
    # 创建更多样化的测试数据
    import trimesh
    import numpy as np
    
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"生成 {num_samples} 个多样化的3D模型...")
    
    # 生成训练集
    for i in range(int(num_samples * 0.8)):
        # 随机选择模型类型
        model_type = random.choice(['sphere', 'box', 'cylinder', 'torus'])
        
        if model_type == 'sphere':
            mesh = trimesh.creation.icosphere(subdivisions=random.randint(1, 3))
        elif model_type == 'box':
            size = np.random.uniform(0.5, 2.0, 3)
            mesh = trimesh.creation.box(extents=size)
        elif model_type == 'cylinder':
            radius = np.random.uniform(0.3, 1.0)
            height = np.random.uniform(0.5, 2.0)
            mesh = trimesh.creation.cylinder(radius=radius, height=height)
        else:  # torus
            mesh = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
        
        # 添加随机变换
        mesh.vertices += np.random.randn(*mesh.vertices.shape) * 0.1
        mesh.export(os.path.join(train_dir, f'{model_type}_{i:05d}.obj'))
    
    # 生成验证集
    for i in range(int(num_samples * 0.2)):
        model_type = random.choice(['sphere', 'box', 'cylinder'])
        
        if model_type == 'sphere':
            mesh = trimesh.creation.icosphere(subdivisions=random.randint(1, 3))
        elif model_type == 'box':
            size = np.random.uniform(0.5, 2.0, 3)
            mesh = trimesh.creation.box(extents=size)
        else:
            radius = np.random.uniform(0.3, 1.0)
            height = np.random.uniform(0.5, 2.0)
            mesh = trimesh.creation.cylinder(radius=radius, height=height)
        
        mesh.vertices += np.random.randn(*mesh.vertices.shape) * 0.1
        mesh.export(os.path.join(val_dir, f'{model_type}_{i:05d}.obj'))
    
    print(f"\n✓ 数据准备完成")
    print(f"训练集: {len(os.listdir(train_dir))} 个文件")
    print(f"验证集: {len(os.listdir(val_dir))} 个文件")


def main():
    parser = argparse.ArgumentParser(description='准备真实训练数据')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='数据根目录')
    parser.add_argument('--method', type=str, 
                       choices=['objaverse', 'local', 'sample'],
                       default='sample',
                       help='数据准备方法: objaverse(从Objaverse下载), local(从本地目录), sample(生成示例数据)')
    parser.add_argument('--source_dir', type=str, default=None,
                       help='本地数据源目录（method=local时使用）')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='样本数量（method=objaverse或sample时使用）')
    
    args = parser.parse_args()
    
    os.makedirs(args.data_root, exist_ok=True)
    
    success = False
    if args.method == 'objaverse':
        success = download_objaverse_data(args.data_root, args.num_samples)
    elif args.method == 'local':
        if not args.source_dir:
            print("错误: 使用 --method local 时需要指定 --source_dir")
            return
        success = prepare_from_directory(args.data_root, args.source_dir)
    elif args.method == 'sample':
        download_sample_data(args.data_root, args.num_samples)
        success = True
    
    if success:
        print("\n" + "=" * 60)
        print("数据准备完成！可以开始训练了。")
        print("运行训练: python train.py --config configs/default.yaml")
        print("=" * 60)
    else:
        print("\n数据准备失败，请检查错误信息。")

if __name__ == "__main__":
    main()

