"""
Blender后处理脚本。

此脚本由blender_postprocess.py调用，在Blender环境中执行。
通过命令行参数接收输入输出路径和配置。
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError:
    print("错误: 此脚本必须在Blender环境中运行")
    sys.exit(1)


def clear_scene():
    """清空场景"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_mesh(filepath: str):
    """
    导入网格文件。

    支持格式: .glb, .obj, .fbx等
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".glb" or suffix == ".gltf":
        bpy.ops.import_scene.gltf(filepath=str(filepath))
    elif suffix == ".obj":
        bpy.ops.import_scene.obj(filepath=str(filepath))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(filepath))
    else:
        raise ValueError(f"不支持的格式: {suffix}")

    # 获取导入的对象
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError("导入失败：未找到网格对象")
    
    return objects[0]


def clean_mesh(obj, config: dict):
    """
    清理和优化网格。

    参数:
        obj: Blender网格对象
        config: 配置字典
    """
    # 进入编辑模式
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    # 1. 选择所有顶点
    bpy.ops.mesh.select_all(action="SELECT")

    # 2. 移除重复顶点（合并距离内的顶点）
    merge_distance = config.get("merge_distance", 0.0001)
    bpy.ops.mesh.remove_doubles(threshold=merge_distance)

    # 3. 删除孤立顶点和边
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)

    # 4. 删除退化面（面积为零的面）
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)

    # 5. 修复非流形边
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=True, use_multi_face=False, use_non_contiguous=False, use_verts=False)
    
    # 删除非流形边
    if bpy.context.selected_edges:
        bpy.ops.mesh.delete(type="EDGE")

    # 6. 修复法线方向
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)

    # 退出编辑模式
    bpy.ops.object.mode_set(mode="OBJECT")

    # 7. 自动填充孔洞
    if config.get("fill_holes", True):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.select_all(action="DESELECT")
        
        # 选择边界边
        bpy.ops.mesh.region_to_loop()
        
        # 填充选中的边界
        if bpy.context.selected_edges:
            bpy.ops.mesh.fill()
        
        bpy.ops.object.mode_set(mode="OBJECT")

    # 8. 初步平滑 - 移除凸起（使用平滑修改器）
    if config.get("preliminary_smooth", True):
        bpy.context.view_layer.objects.active = obj
        smooth_modifier = obj.modifiers.new(name="PreliminarySmooth", type="SMOOTH")
        smooth_modifier.factor = config.get("preliminary_smooth_factor", 0.5)
        smooth_modifier.iterations = config.get("preliminary_smooth_iterations", 1)
        # 应用修改器
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=smooth_modifier.name)

    # 9. 简化模型（使用Decimate修改器）
    if config.get("decimate", True):
        bpy.context.view_layer.objects.active = obj
        decimate_ratio = config.get("decimate_ratio", 0.9)  # 保留90%的几何体
        
        # 计算目标面数
        current_faces = len(obj.data.polygons)
        target_faces = int(current_faces * decimate_ratio)
        
        if target_faces < current_faces and target_faces > 0:
            decimate_modifier = obj.modifiers.new(name="Decimate", type="DECIMATE")
            decimate_modifier.ratio = decimate_ratio
            decimate_modifier.decimate_type = "COLLAPSE"
            # 应用修改器
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)

    # 10. 最终表面平滑
    if config.get("final_smooth", True):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        
        # 使用平滑操作
        smooth_factor = config.get("final_smooth_factor", 0.5)
        smooth_iterations = config.get("final_smooth_iterations", 2)
        bpy.ops.mesh.faces_select_all()
        bpy.ops.mesh.vertices_smooth(
            factor=smooth_factor,
            repeat=smooth_iterations,
        )
        
        bpy.ops.object.mode_set(mode="OBJECT")

    # 最终清理：确保网格是有效的
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def export_mesh(obj, filepath: str):
    """
    导出网格文件。

    支持格式: .glb, .obj, .fbx等
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    # 选择要导出的对象
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)

    if suffix == ".glb" or suffix == ".gltf":
        bpy.ops.export_scene.gltf(
            filepath=str(filepath),
            use_selection=True,
            export_format="GLB" if suffix == ".glb" else "GLTF_SEPARATE",
            export_materials="EXPORT",  # 导出材质和纹理
            export_textures=True,  # 导出纹理
        )
    elif suffix == ".obj":
        bpy.ops.export_scene.obj(
            filepath=str(filepath),
            use_selection=True,
            check_existing=False,
        )
    elif suffix == ".fbx":
        bpy.ops.export_scene.fbx(
            filepath=str(filepath),
            use_selection=True,
        )
    else:
        raise ValueError(f"不支持的导出格式: {suffix}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Blender后处理脚本")
    parser.add_argument("input_path", help="输入网格文件路径")
    parser.add_argument("output_path", help="输出网格文件路径")
    parser.add_argument("config_json", help="配置JSON字符串")
    
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    try:
        # 解析配置
        config = json.loads(args.config_json)

        # 清空场景
        clear_scene()

        # 导入网格
        print(f"[Blender] 导入网格: {args.input_path}")
        obj = import_mesh(args.input_path)
        print(f"[Blender] 导入成功，顶点数: {len(obj.data.vertices)}, 面数: {len(obj.data.polygons)}")

        # 清理和优化
        print("[Blender] 开始后处理...")
        clean_mesh(obj, config)
        print(f"[Blender] 后处理完成，顶点数: {len(obj.data.vertices)}, 面数: {len(obj.data.polygons)}")

        # 导出网格
        print(f"[Blender] 导出网格: {args.output_path}")
        export_mesh(obj, args.output_path)
        print("[Blender] 导出完成")

    except Exception as e:
        print(f"[Blender] 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

