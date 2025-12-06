import os
import torch
import argparse
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

def main():
    parser = argparse.ArgumentParser(description="Generate Multi-View Images using Qwen Image Edit")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image (Front view)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated views")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-Edit-2509", help="Model ID")
    
    args = parser.parse_args()
    
    print(f"--- Qwen Multi-View Generation ---")
    print(f"Input: {args.input_image}")
    print(f"Output Dir: {args.output_dir}")
    
    # 1. Load Model
    print(f"Loading model: {args.model_id}...")
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            args.model_id, 
            torch_dtype=torch.bfloat16
        )
        pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=None)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have installed: pip install git+https://github.com/huggingface/diffusers transformers>=4.51.3 accelerate")
        return

    # 2. Load Image
    if not os.path.exists(args.input_image):
        print(f"Error: Input image {args.input_image} not found.")
        return
    
    image = Image.open(args.input_image).convert("RGB")
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Define Views and Prompts
    # Dictionary mapping view name to prompt
    # Assuming input is "Front"
    
    # Strategy: Use Qwen to rotate the object to get other views.
    # Note: Prompts need to be carefully tuned for best results.
    views_to_generate = {
        "front": None, # Original image
        "left": "Rotate the object 90 degrees clockwise to show the left side",
        "right": "Rotate the object 90 degrees counter-clockwise to show the right side",
        "back": "Rotate the object 180 degrees to show the back side"
    }

    # Save Front View (Original)
    front_path = os.path.join(args.output_dir, "front.png")
    image.save(front_path)
    print(f"Saved Front view to: {front_path}")

    # 4. Generate Other Views
    for view_name, prompt in views_to_generate.items():
        if view_name == "front":
            continue
            
        print(f"Generating {view_name} view with prompt: '{prompt}'...")
        
        inputs = {
            "image": [image], 
            "prompt": prompt,
            "negative_prompt": " ", # Can add negative prompts like "blurred, distorted"
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
            "generator": torch.Generator(device="cuda").manual_seed(42)
        }
        
        with torch.inference_mode():
            output = pipeline(**inputs)
            generated_image = output.images[0]
        
        save_path = os.path.join(args.output_dir, f"{view_name}.png")
        generated_image.save(save_path)
        print(f"Saved {view_name} view to: {save_path}")

    print(f"Success! All views saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()


