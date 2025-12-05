import os
import torch
import argparse
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

def main():
    parser = argparse.ArgumentParser(description="AR Multi-View Generation using Qwen")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image (Front view)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated views")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-Edit-2509", help="Model ID")
    
    args = parser.parse_args()
    
    print(f"--- Qwen AR Multi-View Generation ---")
    print(f"Strategy: Front -> [Front]->Right -> [Front,Right]->Left -> [Front,Right,Left]->Back")
    
    # 1. Load Model
    print(f"Loading model: {args.model_id}...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16
    )
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)

    # 2. Setup Images
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 假设输入图即为参考图/正视图
    # 如果需要单独的参考图，逻辑需微调，这里默认 Input = Front = Ref
    front_img = Image.open(args.input_image).convert("RGB")
    
    # 保存正视图
    front_path = os.path.join(args.output_dir, "front.png")
    front_img.save(front_path)
    print(f"[1/4] Saved Front view")

    generated_views = {
        "front": front_img
    }
    
    # 通用生成参数
    gen_kwargs = {
        "negative_prompt": "distorted, blurry, ugly, low quality",
        "num_inference_steps": 40,
        "guidance_scale": 1.0, # 编辑模型通常较低的 CFG 效果更自然
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device="cuda").manual_seed(42)
    }

    # ==========================================
    # Step 2: Generate Right View
    # Input: [Front]
    # ==========================================
    print(f"[2/4] Generating Right view (Input: Front)...")
    prompt_right = "Rotate the object 90 degrees counter-clockwise to show the right side."
    
    with torch.inference_mode():
        output_right = pipeline(
            image=[front_img], # 单图输入
            prompt=prompt_right,
            **gen_kwargs
        )
    right_img = output_right.images[0]
    right_img.save(os.path.join(args.output_dir, "right.png"))
    generated_views["right"] = right_img

    # ==========================================
    # Step 3: Generate Left View
    # Input: [Front, Right]
    # ==========================================
    print(f"[3/4] Generating Left view (Input: Front + Right)...")
    # 利用右视图的信息来约束左视图（比如对称性、风格）
    prompt_left = "Rotate the object 90 degrees clockwise to show the left side."
    
    with torch.inference_mode():
        output_left = pipeline(
            image=[front_img, right_img], # 多图输入
            prompt=prompt_left,
            **gen_kwargs
        )
    left_img = output_left.images[0]
    left_img.save(os.path.join(args.output_dir, "left.png"))
    generated_views["left"] = left_img

    # ==========================================
    # Step 4: Generate Back View
    # Input: [Front, Right, Left]
    # ==========================================
    print(f"[4/4] Generating Back view (Input: Front + Right + Left)...")
    # 汇聚所有已知信息生成背面
    prompt_back = "Rotate the object 180 degrees to show the back side."
    
    with torch.inference_mode():
        output_back = pipeline(
            image=[front_img, right_img, left_img], # 3图输入
            prompt=prompt_back,
            **gen_kwargs
        )
    back_img = output_back.images[0]
    back_img.save(os.path.join(args.output_dir, "back.png"))
    generated_views["back"] = back_img

    print(f"Success! All AR views saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()


