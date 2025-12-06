import os
import torch
import argparse
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

def main():
    parser = argparse.ArgumentParser(description="Test Qwen Image Edit Only")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Rotate the object 90 degrees clockwise", help="Edit prompt")
    parser.add_argument("--output_path", type=str, default="output_edited.png", help="Path to save edited image")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-Edit-2509", help="Model ID")
    
    args = parser.parse_args()
    
    print(f"--- Qwen Image Edit Test ---")
    print(f"Input: {args.input_image}")
    print(f"Prompt: {args.prompt}")
    
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

    # 3. Edit
    print("Editing...")
    inputs = {
        "image": [image], 
        "prompt": args.prompt,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device="cuda").manual_seed(42)
    }
    
    with torch.inference_mode():
        output = pipeline(**inputs)
        edited_image = output.images[0]
    
    # 4. Save
    edited_image.save(args.output_path)
    print(f"Success! Saved to: {os.path.abspath(args.output_path)}")

if __name__ == "__main__":
    main()


