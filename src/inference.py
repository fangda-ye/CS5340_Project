# src/inference.py
import torch
import numpy as np
from PIL import Image
import os
import random
import argparse
from tqdm.auto import tqdm
import time

# Diffusers imports
from diffusers import StableDiffusionXLPipeline, HunyuanDiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

# Model import
try:
    # Import the refactored NPNet
    from model import NPNet
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure model package is accessible.")
    exit(1)

def set_seed(seed):
    """ Sets random seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # --- Load Base Diffusion Pipeline ---
    print(f"Loading base pipeline: {args.base_model_id}...")
    # Determine pipeline class based on model ID or explicit type
    pipe_class = None
    if "xl" in args.base_model_id.lower() and "hunyuan" not in args.base_model_id.lower():
        pipe_class = StableDiffusionXLPipeline
        print("Assuming SDXL pipeline.")
    elif "hunyuan" in args.base_model_id.lower():
        pipe_class = HunyuanDiTPipeline
        print("Assuming Hunyuan-DiT pipeline.")
    elif "dreamshaper" in args.base_model_id.lower(): # Handle DreamShaper explicitly
            pipe_class = StableDiffusionXLPipeline # Often SDXL based
            print("Assuming SDXL pipeline for DreamShaper.")
    else:
            # Fallback guess based on npnet_model_id
            if args.npnet_model_id == "SDXL" or args.npnet_model_id == "DreamShaper":
                pipe_class = StableDiffusionXLPipeline
            elif args.npnet_model_id == "DiT":
                pipe_class = HunyuanDiTPipeline
            else:
                print(f"Warning: Could not determine pipeline type for {args.base_model_id}. Defaulting to SDXL.")
                pipe_class = StableDiffusionXLPipeline

    try:
        pipe = pipe_class.from_pretrained(
            args.base_model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 and pipe_class == StableDiffusionXLPipeline else None,
            use_safetensors=True,
            low_cpu_mem_usage=True # Reduce CPU RAM usage during loading
        ).to(device)
        # Set scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # Set components to eval mode
        pipe.unet.eval()
        pipe.vae.eval()
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None: pipe.text_encoder.eval()
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None: pipe.text_encoder_2.eval()
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading base pipeline: {e}")
        exit(1)

    # --- Load Trained NPNet Model ---
    print(f"Loading trained NPNet model from: {args.npnet_weights_path}...")
    # Initialize NPNet with parameters matching the trained model
    npnet_model = NPNet(
        model_id=args.npnet_model_id,
        device=device, # NPNet will be moved to device below
        resolution=args.npnet_resolution,
        svd_enable_drop=args.svd_dropout,
        nt_enable_adapter=args.nt_adapter,
        nt_enable_finetune=args.nt_finetune, # Match training config
        nt_enable_dropout=args.nt_dropout,   # Match training config
        enable_cross_attention=args.cross_attention # Match training config
    )
    # Load weights BEFORE moving model to device if loading from full checkpoint
    try:
            # Load weights onto CPU first
            state_dict = torch.load(args.npnet_weights_path, map_location="cpu")
            npnet_model.load_state_dict(state_dict)
    except Exception as e:
            print(f"Error loading NPNet weights: {e}. Ensure the checkpoint matches the model structure.")
            exit(1)

    npnet_model = npnet_model.to(device).to(dtype).eval() # Move to device, set dtype, set to eval mode
    print("NPNet model loaded and set to eval mode.")

    # --- Prepare for Generation ---
    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"Using fixed seed: {args.seed}")
    else:
        # Use a random seed for each generation if seed is None
        current_run_seed = random.randint(0, 2**32 - 1)
        set_seed(current_run_seed)
        generator = torch.Generator(device=device).manual_seed(current_run_seed)
        print(f"Using random seed for this run: {current_run_seed}")
        args.seed = current_run_seed # Store the seed used for filename

    # Calculate latent shape
    vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
    latent_height = args.output_size // vae_scale_factor
    latent_width = args.output_size // vae_scale_factor
    num_channels_latents = pipe.unet.config.in_channels

    # --- Generate Initial Noise ---
    latents_shape = (args.batch_size, num_channels_latents, latent_height, latent_width)
    initial_latents = randn_tensor(
        latents_shape,
        generator=generator,
        device=device,
        dtype=dtype
    )
    print(f"Generated initial noise latents with shape: {initial_latents.shape}")

    # --- Encode Prompt (only needed for NPNet) ---
    # Pipeline call will handle encoding for CFG internally
    print("Encoding prompt for NPNet input...")
    try:
        with torch.no_grad():
            # Encode just the conditional prompt for NPNet
            if args.npnet_model_id == 'SDXL' or args.npnet_model_id == 'DreamShaper':
                prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                    prompt=args.prompt, device=device, num_images_per_prompt=args.batch_size, do_classifier_free_guidance=False
                )
                npnet_prompt_input = prompt_embeds
            elif args.npnet_model_id == 'DiT':
                prompt_embeds = pipe.encode_prompt(prompt=args.prompt, device=device) # Assuming simpler output
                npnet_prompt_input = prompt_embeds
            else:
                raise NotImplementedError(f"Prompt encoding for NPNet not defined for model ID: {args.npnet_model_id}")

            # Ensure batch size matches
            if npnet_prompt_input.shape[0] != args.batch_size:
                npnet_prompt_input = npnet_prompt_input.repeat_interleave(args.batch_size // npnet_prompt_input.shape[0], dim=0)

            print(f"Prompt embeddings for NPNet shape: {npnet_prompt_input.shape}")

    except Exception as e:
        print(f"Error encoding prompt for NPNet: {e}")
        exit(1)


    # --- Generate Golden Noise using NPNet ---
    print("Generating golden noise with NPNet...")
    start_npnet_time = time.time()
    with torch.no_grad():
        # NPNet expects float32 input, convert latents and embeds
        golden_latents = npnet_model(initial_latents.to(torch.float32), npnet_prompt_input.to(torch.float32))
        # Convert output back to inference dtype
        golden_latents = golden_latents.to(dtype)
    end_npnet_time = time.time()
    print(f"Generated golden noise latents with shape: {golden_latents.shape} (took {end_npnet_time - start_npnet_time:.2f}s)")

    # --- Generate Images ---
    common_pipe_args = {
        "prompt": args.prompt, # Pass string prompt
        "height": args.output_size,
        "width": args.output_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": args.batch_size,
        "output_type": "pil"
    }

    images_to_save = {} # Dictionary to store images with descriptive names

    # 1. Generate with Standard Noise (Optional)
    if args.generate_standard:
        print("Generating image with standard noise...")
        if args.seed is not None: # Reset generator only if seed was fixed
            generator = torch.Generator(device=device).manual_seed(args.seed)
        start_std_time = time.time()
        try:
            with torch.no_grad():
                standard_output = pipe(
                    **common_pipe_args,
                    latents=initial_latents,
                    generator=generator
                )
            standard_images = standard_output.images
            end_std_time = time.time()
            print(f"Generated {len(standard_images)} standard image(s) (took {end_std_time - start_std_time:.2f}s).")
            for i, img in enumerate(standard_images):
                images_to_save[f"standard_{i+1}"] = img
        except Exception as e:
            print(f"Error during standard generation: {e}")
    else:
            print("Skipping standard noise generation.")

    # 2. Generate with Golden Noise
    print("Generating image with golden noise...")
    if args.seed is not None: # Reset generator
        generator = torch.Generator(device=device).manual_seed(args.seed)
    start_golden_time = time.time()
    try:
        with torch.no_grad():
            golden_output = pipe(
                **common_pipe_args,
                latents=golden_latents, # Use the golden noise
                generator=generator
            )
        golden_images = golden_output.images
        end_golden_time = time.time()
        print(f"Generated {len(golden_images)} golden image(s) (took {end_golden_time - start_golden_time:.2f}s).")
        for i, img in enumerate(golden_images):
                images_to_save[f"golden_{i+1}"] = img
    except Exception as e:
        print(f"Error during golden generation: {e}")

    # --- Save Images ---
    if not images_to_save:
            print("No images were generated successfully.")
            exit()

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_safe_name = "".join(c if c.isalnum() else "_" for c in args.prompt)[:50]
    seed_suffix = f"_seed{args.seed}" if args.seed is not None else f"_randseed{current_run_seed}" # Use actual random seed if generated

    for name, img in images_to_save.items():
        filename = os.path.join(args.output_dir, f"{prompt_safe_name}{seed_suffix}_{name}.png")
        try:
            img.save(filename)
            print(f"Saved image: {filename}")
        except Exception as e:
            print(f"Error saving image {filename}: {e}")

    print("-" * 100)
    print("Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a trained NPNet model")

    # --- Model Arguments ---
    g_model = parser.add_argument_group('Model Loading')
    g_model.add_argument("--npnet_weights_path", type=str, required=True, help="Path to the trained NPNet weights (.pth file).")
    g_model.add_argument("--npnet_model_id", type=str, default="SDXL", choices=["SDXL", "DreamShaper", "DiT"], help="Specify the base model NPNet was trained for (affects internal setup).")
    g_model.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="HuggingFace ID of the base diffusion model pipeline for generation.")
    g_model.add_argument("--npnet_resolution", type=int, default=128, help="Internal resolution used during NPNet training.")
    # Flags to match NPNet structure during training
    g_model.add_argument("--svd_dropout", action="store_true", help="Flag if SVDNoiseUnet was trained with dropout enabled.")
    g_model.add_argument("--nt_adapter", action="store_true", help="Flag if NoiseTransformer was trained with adapter enabled.")
    g_model.add_argument("--nt_finetune", action="store_true", help="Flag if NoiseTransformer was trained with finetuning enabled.")
    g_model.add_argument("--nt_dropout", action="store_true", default=True, help="Flag if NoiseTransformer was trained with dropout enabled (default: True).")
    g_model.add_argument("--cross_attention", action="store_true", help="Flag if NPNet was trained with cross-attention enabled.")

    # --- Inference Arguments ---
    g_inf = parser.add_argument_group('Inference Parameters')
    g_inf.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    g_inf.add_argument("--output_dir", type=str, default="npnet_output_images", help="Directory to save generated images.")
    g_inf.add_argument("--output_size", type=int, default=1024, help="Output image size (height and width).")
    g_inf.add_argument("--num_inference_steps", type=int, default=30, help="Number of denoising steps for the base model.")
    g_inf.add_argument("--guidance_scale", "--cfg", type=float, default=5.5, help="Classifier-Free Guidance scale.")
    g_inf.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If None, uses a random seed per run.")
    g_inf.add_argument("--batch_size", type=int, default=1, help="Number of images to generate per run.")
    g_inf.add_argument("--generate_standard", action="store_true", help="Also generate an image using the standard initial noise for comparison.")

    # --- System Arguments ---
    g_sys = parser.add_argument_group('System Configuration')
    g_sys.add_argument('--dtype', default='float16', type=str, choices=['float16', 'float32'], help="Data type for inference ('float16' or 'float32').")

    args = parser.parse_args()

    # Adjust default CFG based on model if needed
    if (args.npnet_model_id == 'DreamShaper' or "dreamshaper" in args.base_model_id.lower()) and args.guidance_scale == 5.5:
            args.guidance_scale = 3.5
            print(f"Adjusted CFG scale to {args.guidance_scale} for DreamShaper.")
    elif (args.npnet_model_id == 'DiT' or "hunyuan" in args.base_model_id.lower()) and args.guidance_scale == 5.5:
            args.guidance_scale = 5.0
            print(f"Adjusted CFG scale to {args.guidance_scale} for DiT.")

    main(args)
