# scripts/batch_inference.py
import torch
import numpy as np
from PIL import Image
import os
import random
import argparse
from tqdm.auto import tqdm
import time
import traceback

# Diffusers imports
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

# Model import - Assuming models are in ../model relative to scripts/
import sys
# Add project root to path to allow importing 'model' package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Import the specific RNN model used for training (V3 in this case)
    from model import NoiseSequenceRNN_v3
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure model package and rnn_seq_model_v3.py are accessible from project root.")
    exit(1)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # --- Create Output Directories ---
    output_std_dir = os.path.join(args.output_base_dir, "standard_output")
    output_gns_dir = os.path.join(args.output_base_dir, "gnsnet_output") # As requested
    os.makedirs(output_std_dir, exist_ok=True)
    os.makedirs(output_gns_dir, exist_ok=True)
    print(f"Standard images will be saved to: {output_std_dir}")
    print(f"Golden noise images will be saved to: {output_gns_dir}")

    # --- Load Base Diffusion Pipeline ---
    print(f"Loading base pipeline: {args.base_model_id}...")
    # Simplified pipeline loading assuming SDXL for now
    pipe_class = StableDiffusionXLPipeline
    try:
        pipe = pipe_class.from_pretrained(
            args.base_model_id, torch_dtype=dtype, variant="fp16",
            use_safetensors=True, low_cpu_mem_usage=True
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.unet.eval(); pipe.vae.eval()
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None: pipe.text_encoder.eval()
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None: pipe.text_encoder_2.eval()
        print("Pipeline loaded successfully.")
    except Exception as e: print(f"Error loading base pipeline: {e}"); exit(1)

    # --- Load Trained RNN Sequence Model ---
    print(f"Loading trained RNN sequence model from: {args.rnn_weights_path}...")
    # --- Hardcoded Model Parameters (Adjust if necessary!) ---
    # These should match the parameters used for training the loaded weights
    npnet_model_id_for_config = "SDXL" # Affects text_embed_dim inference if not specified
    text_embed_dim = args.text_embed_dim if args.text_embed_dim > 0 else 1280 # Default for SDXL pooled
    noise_resolution = 128
    noise_channels = 4
    cnn_base_filters = 64
    cnn_num_blocks = [2, 2, 2, 2] # Example, adjust if needed
    cnn_feat_dim = 512
    cnn_groups = 8
    gru_hidden_size = 1024
    gru_num_layers = 2
    predict_variance = True # Assume variance was predicted, adjust if not
    # ---------------------------------------------------------
    rnn_model = NoiseSequenceRNN_v3(
        text_embed_dim=text_embed_dim, noise_img_size=noise_resolution,
        noise_in_chans=noise_channels, cnn_base_filters=cnn_base_filters,
        cnn_num_blocks_per_stage=cnn_num_blocks, cnn_feat_dim=cnn_feat_dim,
        cnn_groups=cnn_groups, gru_hidden_size=gru_hidden_size,
        gru_num_layers=gru_num_layers, gru_dropout=0.0, # No dropout for inference
        predict_variance=predict_variance
    )
    try:
            state_dict = torch.load(args.rnn_weights_path, map_location="cpu")
            rnn_model.load_state_dict(state_dict)
    except Exception as e: print(f"Error loading RNN weights: {e}"); exit(1)
    rnn_model = rnn_model.to(device).to(dtype).eval()
    print("RNN sequence model loaded and set to eval mode.")

    # --- Load Prompts ---
    print(f"Loading prompts from: {args.prompt_file}...")
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts.")
        if not prompts: print("Error: No prompts found in file."); exit(1)
    except Exception as e: print(f"Error reading prompts file: {e}"); exit(1)

    # --- Batch Inference Loop ---
    vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
    latent_height = args.output_size // vae_scale_factor
    latent_width = args.output_size // vae_scale_factor
    num_channels_latents = pipe.unet.config.in_channels
    latents_shape = (1, num_channels_latents, latent_height, latent_width) # Process one prompt at a time (batch=1)

    common_pipe_args = {
        "height": args.output_size, "width": args.output_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "output_type": "pil"
    }

    for idx, prompt in enumerate(tqdm(prompts, desc="Batch Inference")):
        current_seed = args.start_seed + idx
        set_seed(current_seed)
        generator = torch.Generator(device=device).manual_seed(current_seed)

        std_img_path = os.path.join(output_std_dir, f"{idx}.png")
        gns_img_path = os.path.join(output_gns_dir, f"{idx}.png")

        # Skip if both images already exist
        if os.path.exists(std_img_path) and os.path.exists(gns_img_path) and not args.overwrite:
            # print(f"Skipping index {idx}, images already exist.")
            continue

        print(f"\nProcessing prompt {idx}: {prompt[:80]}...")

        try:
            # 1. Generate Initial Noise
            initial_latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)

            # 2. Encode Prompt (for RNN)
            with torch.no_grad():
                # Assuming pooled embedding for RNN V3
                _, _, text_embed, _ = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)

            # 3. Generate Golden Noise Sequence
            with torch.no_grad():
                generated_noise_sequence = rnn_model.generate_sequence(
                    initial_noise=initial_latents.to(dtype),
                    text_embed=text_embed.to(dtype),
                    num_steps=args.num_gen_steps
                )
                final_golden_latent = generated_noise_sequence[:, -1].to(dtype)

            # 4. Generate Standard Image (if not exists or overwrite)
            if not os.path.exists(std_img_path) or args.overwrite:
                print(f"  Generating standard image...")
                set_seed(current_seed); generator = torch.Generator(device=device).manual_seed(current_seed)
                with torch.no_grad():
                    std_output = pipe(prompt=prompt, negative_prompt="", latents=initial_latents, generator=generator, **common_pipe_args)
                std_image = std_output.images[0]
                std_image.save(std_img_path)
                print(f"  Saved standard image: {std_img_path}")
            else:
                    print(f"  Standard image exists, skipping generation.")


            # 5. Generate Golden Noise Image (if not exists or overwrite)
            if not os.path.exists(gns_img_path) or args.overwrite:
                print(f"  Generating golden noise image...")
                set_seed(current_seed); generator = torch.Generator(device=device).manual_seed(current_seed)
                with torch.no_grad():
                    golden_output = pipe(prompt=prompt, negative_prompt="", latents=final_golden_latent, generator=generator, **common_pipe_args)
                golden_image = golden_output.images[0]
                golden_image.save(gns_img_path)
                print(f"  Saved golden noise image: {gns_img_path}")
            else:
                    print(f"  Golden noise image exists, skipping generation.")

        except Exception as e:
            print(f"Error processing prompt {idx} ('{prompt[:50]}...'): {e}")
            traceback.print_exc() # Print full traceback for debugging

        # Optional: Clear cache periodically
        if idx % 20 == 0:
            torch.cuda.empty_cache()

    print("\nBatch inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference using trained RNN Sequence Model")
    # --- Input/Output ---
    parser.add_argument("--prompt_file", type=str, default="./data/pickapic_prompts.txt", help="Path to the text file containing prompts (one per line).")
    parser.add_argument("--output_base_dir", type=str, default="./inference_output/", help="Base directory to save output images (will create 'standard_output' and 'gnsnet_output' subdirs).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images.")
    # --- Model Paths and Config ---
    parser.add_argument("--rnn_weights_path", type=str, default="./output/rnn_v3_seq_model_output/rnn_v3_model_final.pth", help="Path to trained RNN sequence model (.pth).")
    parser.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base diffusion pipeline ID.")
    # --- RNN Model Config (MUST match training) ---
    # These are now hardcoded above for simplicity, but could be args if needed
    parser.add_argument("--text_embed_dim", type=int, default=0, help="Text embedding dim (0 to infer from base_model_id).") # Keep for potential override
    parser.add_argument("--predict_variance", action="store_true", help="Flag if model was trained to predict variance.") # Keep this flag
    # --- Generation Parameters ---
    parser.add_argument("--num_gen_steps", type=int, default=10, help="Number of RNN steps to generate golden noise.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Denoising steps for base model.")
    parser.add_argument("--guidance_scale", "--cfg", type=float, default=5.5, help="CFG scale.")
    parser.add_argument("--output_size", type=int, default=1024, help="Output image size.")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting seed (each prompt gets seed + index).")
    parser.add_argument('--dtype', default='float16', type=str, choices=['float16', 'float32'], help="Inference data type.")

    args = parser.parse_args()
    main(args)
