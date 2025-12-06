import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from diffusers import QwenImageEditPlusPipeline
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置 =================
VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
EDIT_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

class LegacyMVPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vl_model = None
        self.vl_processor = None
        self.edit_pipeline = None
        self.clip_model = None
        self.clip_processor = None

    def load_vl_model(self):
        print(f"[Module 1] Loading VL Model: {VL_MODEL_ID}...")
        self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VL_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.vl_processor = AutoProcessor.from_pretrained(VL_MODEL_ID)

    def unload_vl_model(self):
        if self.vl_model:
            del self.vl_model
            del self.vl_processor
            torch.cuda.empty_cache()
            self.vl_model = None
            print("[Module 1] VL Model unloaded.")

    def load_edit_model(self):
        print(f"[Module 3] Loading Edit Model: {EDIT_MODEL_ID}...")
        self.edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            EDIT_MODEL_ID, torch_dtype=torch.bfloat16
        )
        self.edit_pipeline.to(self.device)

    def unload_edit_model(self):
        if self.edit_pipeline:
            del self.edit_pipeline
            torch.cuda.empty_cache()
            self.edit_pipeline = None
            print("[Module 3] Edit Model unloaded.")

    def load_clip_model(self):
        print(f"[Module 4] Loading CLIP Model: {CLIP_MODEL_ID}...")
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    def analyze_image(self, image_path):
        """Module 1 & 2: View Estimation & Attribute Extraction"""
        if not self.vl_model:
            self.load_vl_model()

        # Prompt Engineering for Analysis
        prompt = (
            "Analyze this image. Provide a JSON output with the following fields:\n"
            "- 'subject': concise name of the main object (e.g., 'red sports car')\n"
            "- 'view': estimation of the current view (e.g., 'front', 'side', 'isometric')\n"
            "- 'attributes': list key visual attributes (color, material, style)\n"
            "- 'description': a detailed caption describing the object's appearance."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.vl_processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.vl_model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vl_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"   [Analysis Result]:\n{output_text}")
        return output_text # In a real system, parse JSON here. For now we use text.

    def construct_prompts(self, analysis_text):
        """Module 2.3: Prompt Construction"""
        # Simple heuristic parsing or use LLM to parse. 
        # For this MVP, we assume analysis_text contains the description.
        # We extract the subject description manually or rely on the whole text.
        
        # To make it robust, we will ask VL to output just the subject description in a second pass 
        # or just use the previous output as context.
        # For simplicity, let's extract a "safe" description prompt.
        
        subject_desc = analysis_text.replace("\n", " ").split("description':")[-1].strip()
        if len(subject_desc) > 200: subject_desc = subject_desc[:200]
        
        prompts = {
            "right": f"Rotate the object 90 degrees counter-clockwise to show the right side. Maintain the style: {subject_desc}",
            "left": f"Rotate the object 90 degrees clockwise to show the left side. Maintain the style: {subject_desc}",
            "back": f"Rotate the object 180 degrees to show the back side. Maintain the style: {subject_desc}",
            "right_45": f"Rotate the object 45 degrees counter-clockwise. Maintain the style: {subject_desc}"
        }
        return prompts

    def generate_candidates(self, image, prompt, k=4):
        """Module 3.2: Candidate Generation"""
        candidates = []
        print(f"   Generating {k} candidates for prompt: '{prompt[:50]}...'")
        
        for i in range(k):
            # Use different seeds for diversity
            generator = torch.Generator(device=self.device).manual_seed(42 + i)
            with torch.inference_mode():
                output = self.edit_pipeline(
                    image=[image],
                    prompt=prompt,
                    negative_prompt="distorted, blurry, low quality, background noise",
                    num_inference_steps=30, # Faster steps for candidates
                    guidance_scale=4.0,
                    generator=generator
                )
            candidates.append(output.images[0])
        return candidates

    def filter_candidates(self, ref_image, candidates, view_name):
        """Module 4: Consistency Filtering using CLIP"""
        if not self.clip_model:
            self.load_clip_model()

        scores = []
        
        # Prepare inputs
        # We compare candidates against the Reference Image (Semantic Consistency)
        try:
            inputs_ref = self.clip_processor(images=ref_image, return_tensors="pt").to(self.device)
            ref_embed = self.clip_model.get_image_features(**inputs_ref)
            ref_embed = ref_embed / ref_embed.norm(p=2, dim=-1, keepdim=True)

            for cand in candidates:
                inputs_cand = self.clip_processor(images=cand, return_tensors="pt").to(self.device)
                cand_embed = self.clip_model.get_image_features(**inputs_cand)
                cand_embed = cand_embed / cand_embed.norm(p=2, dim=-1, keepdim=True)
                
                # Cosine similarity
                score = (ref_embed @ cand_embed.t()).item()
                scores.append(score)
        except Exception as e:
            print(f"CLIP scoring failed: {e}. Selecting first candidate.")
            return candidates[0], 0.0

        best_idx = np.argmax(scores)
        print(f"   [Filter] View: {view_name} | Best Score: {scores[best_idx]:.4f} (Index {best_idx})")
        return candidates[best_idx], scores[best_idx]

    def run(self, input_path, output_dir, k_candidates=2):
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Stage 0 & 1: Analysis ---
        analysis = self.analyze_image(input_path)
        self.unload_vl_model() # Save VRAM

        # --- Stage 2: Prompting ---
        prompts = self.construct_prompts(analysis)
        
        # --- Stage 3: Generation ---
        self.load_edit_model()
        input_img = Image.open(input_path).convert("RGB")
        input_img.save(os.path.join(output_dir, "front.png"))
        
        results = {"front": input_img}
        
        # Define view strategy (AR style can be integrated here, but let's stick to parallel for Legacy MVP)
        target_views = ["right", "left", "back"]
        
        current_ref = input_img # For AR, this would update. For now, reference is always Front.
        
        for view in target_views:
            prompt = prompts[view]
            
            # Generate K candidates
            candidates = self.generate_candidates(current_ref, prompt, k=k_candidates)
            
            # --- Stage 4: Filtering ---
            # We unload edit model temporarily if we need massive VRAM for CLIP, 
            # but usually CLIP + Edit model can fit in 80G/40G. 
            # If OOM, we would need to juggle models.
            
            best_img, score = self.filter_candidates(input_img, candidates, view)
            
            save_path = os.path.join(output_dir, f"{view}.png")
            best_img.save(save_path)
            results[view] = best_img
            
            # Optional: Update reference for next view (Simple AR)
            # current_ref = best_img 

        self.unload_edit_model()
        print(f"Done! Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--candidates", type=int, default=2, help="Number of candidates per view")
    args = parser.parse_args()
    
    pipeline = LegacyMVPipeline()
    pipeline.run(args.input_image, args.output_dir, args.candidates)

if __name__ == "__main__":
    main()

