# src/inference_rnn.py
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
from diffusers import StableDiffusionXLPipeline, HunyuanDiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

# Model import
try:
    # Import the specific RNN model used for training (V3 in this case)
    from model import NoiseSequenceRNN_v3
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure model package and rnn_seq_model_v3.py are accessible.")
    exit(1)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # --- Load Base Diffusion Pipeline ---
    print(f"Loading base pipeline: {args.base_model_id}...")
    # (Same pipeline loading logic as train script)
    pipe_class = None
    if "xl" in args.base_model_id.lower() and "hunyuan" not in args.base_model_id.lower(): pipe_class = StableDiffusionXLPipeline
    elif "hunyuan" in args.base_model_id.lower(): pipe_class = HunyuanDiTPipeline
    elif "dreamshaper" in args.base_model_id.lower(): pipe_class = StableDiffusionXLPipeline
    else: pipe_class = StableDiffusionXLPipeline # Default guess

    try:
        pipe = pipe_class.from_pretrained(
            args.base_model_id, torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 and pipe_class == StableDiffusionXLPipeline else None,
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
    # Determine text_embed_dim based on base model (same logic as training)
    if args.npnet_model_id == 'DiT': text_embed_dim = 1024
    else: text_embed_dim = 1280 # SDXL pooled
    if args.text_embed_dim > 0: text_embed_dim = args.text_embed_dim

    # Initialize model with parameters matching the trained one
    rnn_model = NoiseSequenceRNN_v3(
        text_embed_dim=text_embed_dim, noise_img_size=args.noise_resolution,
        noise_in_chans=args.noise_channels, cnn_base_filters=args.cnn_base_filters,
        cnn_num_blocks_per_stage=args.cnn_num_blocks, cnn_feat_dim=args.cnn_feat_dim,
        cnn_groups=args.cnn_groups, gru_hidden_size=args.gru_hidden_size,
        gru_num_layers=args.gru_num_layers, gru_dropout=0.0, # No dropout during inference
        predict_variance=args.predict_variance # Must match trained model
    )
    try:
            state_dict = torch.load(args.rnn_weights_path, map_location="cpu")
            rnn_model.load_state_dict(state_dict)
    except Exception as e: print(f"Error loading RNN weights: {e}"); exit(1)
    rnn_model = rnn_model.to(device).to(dtype).eval() # Move to device, set dtype, set to eval
    print("RNN sequence model loaded and set to eval mode.")

    # --- Prepare for Generation ---
    if args.seed is not None: set_seed(args.seed); generator = torch.Generator(device=device).manual_seed(args.seed); print(f"Using fixed seed: {args.seed}")
    else: current_run_seed = random.randint(0, 2**32 - 1); set_seed(current_run_seed); generator = torch.Generator(device=device).manual_seed(current_run_seed); print(f"Using random seed: {current_run_seed}"); args.seed = current_run_seed

    vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
    latent_height = args.output_size // vae_scale_factor; latent_width = args.output_size // vae_scale_factor
    num_channels_latents = pipe.unet.config.in_channels
    latents_shape = (args.batch_size, num_channels_latents, latent_height, latent_width)

    # --- Generate Initial Noise ---
    initial_latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)
    print(f"Generated initial noise latents with shape: {initial_latents.shape}")

    # --- Encode Prompt (for RNN model) ---
    print("Encoding prompt for RNN input...")
    try:
        with torch.no_grad():
            # Get the pooled embedding (or adapt encode_text if using sequence)
            if "xl" in args.base_model_id.lower() and "hunyuan" not in args.base_model_id.lower():
                    _, _, text_embed, _ = pipe.encode_prompt(prompt=args.prompt, device=device, num_images_per_prompt=args.batch_size, do_classifier_free_guidance=False)
            elif "hunyuan" in args.base_model_id.lower():
                    text_embed = pipe.encode_prompt(prompt=args.prompt, device=device) # Adjust based on actual output
            else: # Default/Fallback
                    _, _, text_embed, _ = pipe.encode_prompt(prompt=args.prompt, device=device, num_images_per_prompt=args.batch_size, do_classifier_free_guidance=False)

            # Ensure batch size matches
            if text_embed.shape[0] != args.batch_size:
                text_embed = text_embed.repeat_interleave(args.batch_size // text_embed.shape[0], dim=0)
            print(f"Text embedding shape for RNN: {text_embed.shape}")
    except Exception as e: print(f"Error encoding prompt for RNN: {e}"); exit(1)

    # --- Generate Golden Noise Sequence using RNN ---
    print(f"Generating golden noise sequence ({args.num_gen_steps} steps) with RNN...")
    start_rnn_time = time.time()
    with torch.no_grad():
        # Use the model's generate_sequence method
        # Ensure inputs are float32 if model expects it internally during generation
        generated_noise_sequence = rnn_model.generate_sequence(
            initial_noise=initial_latents.to(torch.float32),
            text_embed=text_embed.to(torch.float32),
            num_steps=args.num_gen_steps
        )
        # Select the final noise state and convert to inference dtype
        final_golden_latent = generated_noise_sequence[:, -1].to(dtype) # Get last noise [B, C, H, W]
    end_rnn_time = time.time()
    print(f"Generated final golden noise latent with shape: {final_golden_latent.shape} (took {end_rnn_time - start_rnn_time:.2f}s)")

    # --- Generate Images ---
    common_pipe_args = {
        "prompt": args.prompt, "height": args.output_size, "width": args.output_size,
        "num_inference_steps": args.num_inference_steps, "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": args.batch_size, "output_type": "pil"
    }
    images_to_save = {}

    # 1. Generate with Standard Noise (Optional)
    if args.generate_standard:
        print("Generating image with standard noise...")
        if args.seed is not None: generator = torch.Generator(device=device).manual_seed(args.seed)
        start_std_time = time.time()
        try:
            with torch.no_grad(): std_output = pipe(latents=initial_latents, generator=generator, **common_pipe_args)
            for i, img in enumerate(std_output.images): images_to_save[f"standard_{i+1}"] = img
            end_std_time = time.time(); print(f"Standard generation took {end_std_time - start_std_time:.2f}s.")
        except Exception as e: print(f"Error during standard generation: {e}")
    else: print("Skipping standard noise generation.")

    # 2. Generate with Final Golden Noise from RNN
    print("Generating image with final golden noise from RNN...")
    if args.seed is not None: generator = torch.Generator(device=device).manual_seed(args.seed)
    start_golden_time = time.time()
    try:
        with torch.no_grad(): golden_output = pipe(latents=final_golden_latent, generator=generator, **common_pipe_args)
        for i, img in enumerate(golden_output.images): images_to_save[f"rnn_golden_{i+1}"] = img
        end_golden_time = time.time(); print(f"Golden generation took {end_golden_time - start_golden_time:.2f}s.")
    except Exception as e: print(f"Error during golden generation: {e}")

    # --- Save Images ---
    if not images_to_save: print("No images generated."); exit()
    os.makedirs(args.output_dir, exist_ok=True)
    prompt_safe_name = "".join(c if c.isalnum() else "_" for c in args.prompt)[:50]
    seed_suffix = f"_seed{args.seed}" if args.seed is not None else f"_randseed{current_run_seed}"
    for name, img in images_to_save.items():
        filename = os.path.join(args.output_dir, f"{prompt_safe_name}{seed_suffix}_{name}.png")
        try: img.save(filename); print(f"Saved image: {filename}")
        except Exception as e: print(f"Error saving image {filename}: {e}")
    print("-" * 50); print("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a trained RNN Sequence Model")
    # --- Model Arguments ---
    g_model = parser.add_argument_group('Model Loading')
    g_model.add_argument("--rnn_weights_path", type=str, required=True, help="Path to trained RNN sequence model (.pth).")
    g_model.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base diffusion pipeline ID.")
    g_model.add_argument("--npnet_model_id", type=str, default="SDXL", choices=["SDXL", "DiT"], help="Base model type hint for text dim.")
    g_model.add_argument("--text_embed_dim", type=int, default=0, help="Dimension of text embeddings (0 to infer).")
    g_model.add_argument("--noise_resolution", type=int, default=128, help="Noise spatial resolution.")
    g_model.add_argument("--noise_channels", type=int, default=4, help="Noise channels.")
    g_model.add_argument("--cnn_base_filters", type=int, default=64, help="CNN base filters.")
    g_model.add_argument("--cnn_num_blocks", type=int, nargs='+', default=[2, 2, 2, 2], help="List: CNN ResBlocks per stage.")
    g_model.add_argument("--cnn_feat_dim", type=int, default=512, help="CNN feature dimension.")
    g_model.add_argument("--cnn_groups", type=int, default=8, help="GroupNorm groups.")
    g_model.add_argument("--gru_hidden_size", type=int, default=1024, help="GRU hidden size.")
    g_model.add_argument("--gru_num_layers", type=int, default=2, help="GRU layers.")
    g_model.add_argument("--predict_variance", action="store_true", help="Flag if model was trained to predict variance.")
    # --- Inference Arguments ---
    g_inf = parser.add_argument_group('Inference Parameters')
    g_inf.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    g_inf.add_argument("--output_dir", type=str, default="rnn_inference_output", help="Output directory.")
    g_inf.add_argument("--output_size", type=int, default=1024, help="Output image size.")
    g_inf.add_argument("--num_inference_steps", type=int, default=30, help="Denoising steps for base model.")
    g_inf.add_argument("--guidance_scale", "--cfg", type=float, default=5.5, help="CFG scale.")
    g_inf.add_argument("--seed", type=int, default=None, help="Random seed (None for random).")
    g_inf.add_argument("--batch_size", type=int, default=1, help="Number of images to generate.")
    g_inf.add_argument("--num_gen_steps", type=int, default=10, help="Number of RNN steps to generate golden noise.")
    g_inf.add_argument("--generate_standard", action="store_true", help="Also generate standard noise image.")
    # --- System Arguments ---
    g_sys = parser.add_argument_group('System Configuration')
    g_sys.add_argument('--dtype', default='float16', type=str, choices=['float16', 'float32'], help="Inference data type.")

    args = parser.parse_args()
    if isinstance(args.cnn_num_blocks, list) and len(args.cnn_num_blocks) > 0 and isinstance(args.cnn_num_blocks[0], str):
            try: args.cnn_num_blocks = [int(b) for b in args.cnn_num_blocks]
            except ValueError: print("Error: --cnn_num_blocks must be integers."); exit(1)
    main(args)
