import os
import torch
import argparse
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def get_caption(image_path, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
    """
    Use Qwen2.5-VL to generate a concise caption for the image.
    """
    print(f"--- Step 0: Generating Caption with {model_id} ---")
    
    try:
        # Load VL Model
        # Qwen2.5-VL-3B is small enough to load quickly
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Prepare Input
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "Describe the main object in this image in detail. Focus on its type, color, material, and pose. If it is an animal or creature, explicitly state how many legs it has and its standing/sitting posture. Keep it as a single detailed paragraph."},
                ],
            }
        ]
        
        # Inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Increase max_new_tokens to allow for detailed description
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        caption = output_text[0].strip()
        
        print(f"   [√] Generated Caption: '{caption}'")
        
        # Free VRAM immediately!
        del model
        del processor
        torch.cuda.empty_cache()
        
        return caption

    except Exception as e:
        print(f"   [!] Warning: Caption generation failed ({e}). Using default 'object'.")
        return "object"

def main():
    parser = argparse.ArgumentParser(description="AR Multi-View Generation using Qwen (with VL Captioning)")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--vl_model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="VL Model ID")
    parser.add_argument("--edit_model_id", type=str, default="Qwen/Qwen-Image-Edit-2509", help="Edit Model ID")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "simple_45"], 
                        help="Generation mode: 'full' (4 views) or 'simple_45' (front + 45 deg)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Generate Caption (Subject Recognition)
    # We do this FIRST to get the "subject"
    subject = get_caption(args.input_image, args.vl_model_id)
    
    # 2. Load Edit Model
    print(f"--- Step 1: Loading Edit Model {args.edit_model_id} ---")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.edit_model_id, 
        torch_dtype=torch.bfloat16
    )
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)

    # 3. Prepare Images
    front_img = Image.open(args.input_image).convert("RGB")
    front_path = os.path.join(args.output_dir, "front.png")
    front_img.save(front_path)
    
    gen_kwargs = {
        "negative_prompt": "distorted, blurry, ugly, low quality, background, noise",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device="cuda").manual_seed(42)
    }

    print(f"--- Step 2: Generating Views for '{subject}' (Mode: {args.mode}) ---")

    if args.mode == "simple_45":
        # === Simple 45 Degree Mode ===
        # Generate ONLY one view rotated by 45 degrees
        prompt_45 = f"Rotate the {subject} 45 degrees counter-clockwise to show a perspective view."
        print(f"   Generating 45-deg View: '{prompt_45}'")
        
        with torch.inference_mode():
            out_45 = pipeline(image=[front_img], prompt=prompt_45, **gen_kwargs)
        
        img_45 = out_45.images[0]
        # Save as 'right_45.png' to distinguish
        img_45.save(os.path.join(args.output_dir, "right_45.png"))
        print(f"   [√] Saved 45-deg view to: {os.path.join(args.output_dir, 'right_45.png')}")

    else:
        # === Full 4-View AR Mode (Original) ===
        # [2/4] Right View
        prompt_right = f"Rotate the {subject} 90 degrees counter-clockwise to show the right side."
        print(f"   Generating Right: '{prompt_right}'")
        with torch.inference_mode():
            out_right = pipeline(image=[front_img], prompt=prompt_right, **gen_kwargs)
        right_img = out_right.images[0]
        right_img.save(os.path.join(args.output_dir, "right.png"))

        # [3/4] Left View (Input: Front + Right)
        prompt_left = f"Rotate the {subject} 90 degrees clockwise to show the left side."
        print(f"   Generating Left: '{prompt_left}'")
        with torch.inference_mode():
            out_left = pipeline(image=[front_img, right_img], prompt=prompt_left, **gen_kwargs)
        left_img = out_left.images[0]
        left_img.save(os.path.join(args.output_dir, "left.png"))

        # [4/4] Back View (Input: Front + Right + Left)
        prompt_back = f"Rotate the {subject} 180 degrees to show the back side."
        print(f"   Generating Back: '{prompt_back}'")
        with torch.inference_mode():
            out_back = pipeline(image=[front_img, right_img, left_img], prompt=prompt_back, **gen_kwargs)
        back_img = out_back.images[0]
        back_img.save(os.path.join(args.output_dir, "back.png"))

    print(f"Success! All views saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()

