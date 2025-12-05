import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from PIL import Image

# Add the core directory to sys.path to allow imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.pipeline import QwenEditMVPipeline

def main():
    parser = argparse.ArgumentParser(description="Test Qwen Edit -> MV Generation Pipeline")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Rotate the object 90 degrees clockwise", help="Edit prompt for Qwen")
    parser.add_argument("--output_name", type=str, default="qwen_mv_test", help="Output asset name")
    parser.add_argument("--config", type=str, default="configs/hunyuan3d_mv.yaml", help="Path to MV config")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # 1. Check Input
    if not os.path.exists(args.input_image):
        print(f"Error: Input image {args.input_image} not found.")
        return
        
    # 2. Load Config
    print(f"Loading config from {args.config}...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override device if needed
    config["device"] = args.device
    
    # Ensure output dir exists
    config["default_output_dir"] = "outputs/qwen_tests"
    os.makedirs(config["default_output_dir"], exist_ok=True)

    # 3. Initialize Pipeline
    # Note: This will load the Hunyuan3D model weights specified in the config
    print("Initializing QwenEditMVPipeline...")
    try:
        pipeline = QwenEditMVPipeline(config)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        print("Make sure you have the correct weights and environment.")
        return

    # 4. Run Edit + Generate
    print(f"Running Edit + Generate...")
    print(f"Input: {args.input_image}")
    print(f"Prompt: {args.prompt}")
    
    try:
        result = pipeline.edit_and_generate(
            image_path=args.input_image,
            edit_prompt=args.prompt,
            output_name=args.output_name,
            file_format="glb"
        )
        
        print("-" * 50)
        print("Success!")
        print(f"Edited Image saved at: {result.get('edited_image')}") # Note: logic in pipeline saves it, result has the object
        print(f"3D Asset saved at: {result.get('mesh_path')}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


