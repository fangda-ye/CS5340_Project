# src/evaluate.py
import torch
import numpy as np
from PIL import Image
import os
import random
import argparse
from tqdm.auto import tqdm
import time
import pandas as pd

# Diffusers imports
from diffusers import StableDiffusionXLPipeline, HunyuanDiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

# Model import
try:
    from model import NPNet
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure model package is accessible.")
    exit(1)

# HPSv2 import (and potentially others later)
try:
    import hpsv2
    HPSV2_AVAILABLE = True
except ImportError:
    print("Warning: hpsv2 library not found. HPSv2 metric calculation will be skipped.")
    HPSV2_AVAILABLE = False

# Placeholder for other metrics (e.g., CLIPScore, ImageReward)
# try:
#     from clip_score import calculate_clip_score # Example
#     CLIP_SCORE_AVAILABLE = True
# except ImportError:
#     CLIP_SCORE_AVAILABLE = False
CLIP_SCORE_AVAILABLE = False # Disable for now
IMAGE_REWARD_AVAILABLE = False # Disable for now


def set_seed(seed):
    """ Sets random seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(image_std, image_golden, prompt, args):
    """ Calculates requested evaluation metrics. """
    results = {}
    # HPSv2
    if HPSV2_AVAILABLE and 'hpsv2' in args.metrics:
        try:
            score_std = hpsv2.score(image_std, prompt, hps_version=args.hps_version)[0]
            score_golden = hpsv2.score(image_golden, prompt, hps_version=args.hps_version)[0]
            results['hpsv2_std'] = score_std
            results['hpsv2_golden'] = score_golden
            results['hpsv2_diff'] = score_golden - score_std
        except Exception as e:
            print(f"Warning: HPSv2 calculation failed for prompt '{prompt[:30]}...': {e}")
            results['hpsv2_std'] = np.nan
            results['hpsv2_golden'] = np.nan
            results['hpsv2_diff'] = np.nan
    else:
            results['hpsv2_std'] = np.nan
            results['hpsv2_golden'] = np.nan
            results['hpsv2_diff'] = np.nan


    # CLIP Score (Placeholder)
    if CLIP_SCORE_AVAILABLE and 'clipscore' in args.metrics:
        # Needs implementation: Load CLIP model, calculate score
        # score_std = calculate_clip_score(image_std, prompt)
        # score_golden = calculate_clip_score(image_golden, prompt)
        # results['clipscore_std'] = score_std
        # results['clipscore_golden'] = score_golden
        # results['clipscore_diff'] = score_golden - score_std
        pass
    else:
            results['clipscore_std'] = np.nan
            results['clipscore_golden'] = np.nan
            results['clipscore_diff'] = np.nan

    # ImageReward (Placeholder)
    if IMAGE_REWARD_AVAILABLE and 'imagereward' in args.metrics:
        # Needs implementation: Load ImageReward model, calculate score
        # score_std = calculate_imagereward(image_std, prompt)
        # score_golden = calculate_imagereward(image_golden, prompt)
        # results['imagereward_std'] = score_std
        # results['imagereward_golden'] = score_golden
        # results['imagereward_diff'] = score_golden - score_std
        pass
    else:
            results['imagereward_std'] = np.nan
            results['imagereward_golden'] = np.nan
            results['imagereward_diff'] = np.nan

    return results


def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")
    os.makedirs(args.output_dir, exist_ok=True)
    img_save_dir = os.path.join(args.output_dir, "images")
    if args.save_images:
            os.makedirs(img_save_dir, exist_ok=True)

    # --- Load Base Diffusion Pipeline ---
    print(f"Loading base pipeline: {args.base_model_id}...")
    # Determine pipeline class (same logic as inference.py)
    pipe_class = None
    if "xl" in args.base_model_id.lower() and "hunyuan" not in args.base_model_id.lower(): pipe_class = StableDiffusionXLPipeline
    elif "hunyuan" in args.base_model_id.lower(): pipe_class = HunyuanDiTPipeline
    elif "dreamshaper" in args.base_model_id.lower(): pipe_class = StableDiffusionXLPipeline
    else:
            if args.npnet_model_id == "SDXL" or args.npnet_model_id == "DreamShaper": pipe_class = StableDiffusionXLPipeline
            elif args.npnet_model_id == "DiT": pipe_class = HunyuanDiTPipeline
            else: pipe_class = StableDiffusionXLPipeline

    try:
        pipe = pipe_class.from_pretrained(
            args.base_model_id, torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 and pipe_class == StableDiffusionXLPipeline else None,
            use_safetensors=True, low_cpu_mem_usage=True
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
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
    npnet_model = NPNet(
        model_id=args.npnet_model_id, device=device, resolution=args.npnet_resolution,
        svd_enable_drop=args.svd_dropout, nt_enable_adapter=args.nt_adapter,
        nt_enable_finetune=args.nt_finetune, nt_enable_dropout=args.nt_dropout,
        enable_cross_attention=args.cross_attention
    )
    try:
            state_dict = torch.load(args.npnet_weights_path, map_location="cpu")
            npnet_model.load_state_dict(state_dict)
    except Exception as e:
            print(f"Error loading NPNet weights: {e}")
            exit(1)
    npnet_model = npnet_model.to(device).to(dtype).eval()
    print("NPNet model loaded and set to eval mode.")

    # --- Load Evaluation Prompts ---
    print(f"Loading evaluation prompts from: {args.evaluation_prompts_file}...")
    try:
        if args.evaluation_prompts_file.lower().endswith(".csv"):
            df_prompts = pd.read_csv(args.evaluation_prompts_file)
            if args.prompt_column not in df_prompts.columns:
                raise ValueError(f"Column '{args.prompt_column}' not found in CSV.")
            prompts = df_prompts[args.prompt_column].astype(str).tolist()
        elif args.evaluation_prompts_file.lower().endswith(".txt"):
            with open(args.evaluation_prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
                raise ValueError("Unsupported prompt file format. Use .txt or .csv")
        prompts = [p for p in prompts if isinstance(p, str) and p]
        print(f"Loaded {len(prompts)} valid prompts for evaluation.")
        if not prompts:
                print("Error: No valid prompts found.")
                exit(1)
        # Limit prompts if max_prompts is set
        if args.max_prompts is not None and args.max_prompts > 0:
                prompts = prompts[:args.max_prompts]
                print(f"Evaluating on the first {len(prompts)} prompts.")

    except Exception as e:
        print(f"Error loading or parsing prompts file: {e}")
        exit(1)

    # --- Evaluation Loop ---
    results_list = []
    vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
    latent_height = args.output_size // vae_scale_factor
    latent_width = args.output_size // vae_scale_factor
    num_channels_latents = pipe.unet.config.in_channels
    latents_shape = (args.batch_size, num_channels_latents, latent_height, latent_width)

    common_pipe_args = {
        "height": args.output_size, "width": args.output_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": args.batch_size, # Should be 1 for eval usually
        "output_type": "pil"
    }
    if args.batch_size != 1:
        print("Warning: Batch size > 1 for evaluation. Metrics might be averaged or only calculated for the first image.")


    for i, prompt in enumerate(tqdm(prompts, desc="Evaluating Prompts")):
        eval_seed = args.seed + i if args.seed is not None else random.randint(0, 2**32 - 1)
        set_seed(eval_seed)
        generator = torch.Generator(device=device).manual_seed(eval_seed)

        # --- Generate Noise ---
        initial_latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)

        # --- Encode Prompt for NPNet ---
        try:
            with torch.no_grad():
                if args.npnet_model_id == 'SDXL' or args.npnet_model_id == 'DreamShaper':
                    prompt_embeds, _, _, _ = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=args.batch_size, do_classifier_free_guidance=False)
                elif args.npnet_model_id == 'DiT':
                    prompt_embeds = pipe.encode_prompt(prompt=prompt, device=device)
                else: raise NotImplementedError
                if prompt_embeds.shape[0] != args.batch_size:
                    prompt_embeds = prompt_embeds.repeat_interleave(args.batch_size // prompt_embeds.shape[0], dim=0)
        except Exception as e:
            print(f"Error encoding prompt '{prompt[:30]}...': {e}. Skipping.")
            continue

        # --- Generate Golden Noise ---
        try:
            with torch.no_grad():
                golden_latents = npnet_model(initial_latents.to(torch.float32), prompt_embeds.to(torch.float32)).to(dtype)
        except Exception as e:
                print(f"Error during NPNet forward pass for prompt '{prompt[:30]}...': {e}. Skipping.")
                continue

        # --- Generate Images ---
        image_std, image_golden = None, None
        # Standard Image
        set_seed(eval_seed) # Reset seed
        generator = torch.Generator(device=device).manual_seed(eval_seed)
        try:
            with torch.no_grad():
                std_output = pipe(prompt=prompt, latents=initial_latents, generator=generator, **common_pipe_args)
            image_std = std_output.images[0] # Get first image if batch_size > 1
        except Exception as e:
            print(f"Error generating standard image for prompt '{prompt[:30]}...': {e}")

        # Golden Image
        set_seed(eval_seed) # Reset seed
        generator = torch.Generator(device=device).manual_seed(eval_seed)
        try:
            with torch.no_grad():
                golden_output = pipe(prompt=prompt, latents=golden_latents, generator=generator, **common_pipe_args)
            image_golden = golden_output.images[0] # Get first image
        except Exception as e:
            print(f"Error generating golden image for prompt '{prompt[:30]}...': {e}")

        # --- Calculate Metrics ---
        if image_std and image_golden:
            metric_results = calculate_metrics(image_std, image_golden, prompt, args)
        else:
            # Create dummy results if generation failed
            metric_results = {m+'_std': np.nan for m in args.metrics}
            metric_results.update({m+'_golden': np.nan for m in args.metrics})
            metric_results.update({m+'_diff': np.nan for m in args.metrics})


        # --- Store Results ---
        prompt_results = {
            "index": i,
            "prompt": prompt,
            "seed": eval_seed,
            **metric_results # Add all calculated metric scores
        }
        results_list.append(prompt_results)

        # --- Save Images (Optional) ---
        if args.save_images:
                prompt_safe_name = "".join(c if c.isalnum() else "_" for c in prompt)[:50]
                seed_suffix = f"_seed{eval_seed}"
                if image_std:
                    img_std_path = os.path.join(img_save_dir, f"{prompt_safe_name}{seed_suffix}_standard.png")
                    image_std.save(img_std_path)
                if image_golden:
                    img_golden_path = os.path.join(img_save_dir, f"{prompt_safe_name}{seed_suffix}_golden.png")
                    image_golden.save(img_golden_path)

        # Optional: Clear cache periodically
        if i % 20 == 0:
                torch.cuda.empty_cache()

    # --- Aggregate and Save Results ---
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(args.output_dir, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved detailed evaluation results to: {results_path}")

    # Calculate and print average scores
    print("\n--- Evaluation Summary ---")
    print(f"Evaluated {len(results_df)} prompts.")
    for metric in args.metrics:
        avg_std = results_df[f'{metric}_std'].mean()
        avg_golden = results_df[f'{metric}_golden'].mean()
        avg_diff = results_df[f'{metric}_diff'].mean()
        print(f"Average {metric.upper()}:")
        print(f"  - Standard: {avg_std:.4f}")
        print(f"  - Golden:   {avg_golden:.4f}")
        print(f"  - Difference (Golden - Standard): {avg_diff:.4f}")

    print("-" * 26)
    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained NPNet model")

    # --- Model Arguments ---
    g_model = parser.add_argument_group('Model Loading')
    g_model.add_argument("--npnet_weights_path", type=str, required=True, help="Path to the trained NPNet weights (.pth file).")
    g_model.add_argument("--npnet_model_id", type=str, default="SDXL", choices=["SDXL", "DreamShaper", "DiT"], help="Base model type NPNet was trained for.")
    g_model.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base diffusion pipeline ID for generation.")
    g_model.add_argument("--npnet_resolution", type=int, default=128, help="Internal resolution used during NPNet training.")
    # Flags matching NPNet structure
    g_model.add_argument("--svd_dropout", action="store_true", help="Flag if SVDNoiseUnet was trained with dropout.")
    g_model.add_argument("--nt_adapter", action="store_true", help="Flag if NoiseTransformer was trained with adapter.")
    g_model.add_argument("--nt_finetune", action="store_true", help="Flag if NoiseTransformer was trained with finetuning.")
    g_model.add_argument("--nt_dropout", action="store_true", default=True, help="Flag if NoiseTransformer was trained with dropout.")
    g_model.add_argument("--cross_attention", action="store_true", help="Flag if NPNet was trained with cross-attention.")

    # --- Evaluation Arguments ---
    g_eval = parser.add_argument_group('Evaluation Parameters')
    g_eval.add_argument("--evaluation_prompts_file", type=str, required=True, help="Path to evaluation prompts file (.txt or .csv).")
    g_eval.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompts if using CSV.")
    g_eval.add_argument("--output_dir", type=str, default="npnet_evaluation_output", help="Directory for results and optional images.")
    g_eval.add_argument("--output_size", type=int, default=1024, help="Output image size.")
    g_eval.add_argument("--num_inference_steps", type=int, default=30, help="Denoising steps for base model.")
    g_eval.add_argument("--guidance_scale", "--cfg", type=float, default=5.5, help="CFG scale.")
    g_eval.add_argument("--seed", type=int, default=None, help="Fixed starting seed for evaluation (prompts will iterate: seed, seed+1, ...). If None, uses random seeds.")
    g_eval.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (recommend 1 for evaluation).")
    g_eval.add_argument("--max_prompts", type=int, default=None, help="Maximum number of prompts to evaluate from the file.")
    g_eval.add_argument("--save_images", action="store_true", help="Save generated standard and golden images.")
    g_eval.add_argument("--metrics", nargs='+', default=['hpsv2'], help="List of metrics to calculate (e.g., hpsv2 clipscore imagereward).")
    g_eval.add_argument("--hps_version", type=str, default="v2.1", choices=["v2.0", "v2.1"], help="Version of HPSv2 model to use.")


    # --- System Arguments ---
    g_sys = parser.add_argument_group('System Configuration')
    g_sys.add_argument('--dtype', default='float16', type=str, choices=['float16', 'float32'], help="Data type for inference.")

    args = parser.parse_args()

    # Validate metrics
    supported_metrics = ['hpsv2'] # Add 'clipscore', 'imagereward' when implemented
    args.metrics = [m.lower() for m in args.metrics]
    for m in args.metrics:
        if m not in supported_metrics:
            print(f"Warning: Metric '{m}' is not currently supported/implemented. Supported: {supported_metrics}")
    # Filter to only supported metrics for now
    args.metrics = [m for m in args.metrics if m in supported_metrics]
    if not args.metrics:
            print("Error: No supported metrics selected for evaluation.")
            # exit(1) # Or just run generation without metrics

    # Adjust default CFG based on model if needed
    if (args.npnet_model_id == 'DreamShaper' or "dreamshaper" in args.base_model_id.lower()) and args.guidance_scale == 5.5:
            args.guidance_scale = 3.5
            print(f"Adjusted CFG scale to {args.guidance_scale} for DreamShaper.")
    elif (args.npnet_model_id == 'DiT' or "hunyuan" in args.base_model_id.lower()) and args.guidance_scale == 5.5:
            args.guidance_scale = 5.0
            print(f"Adjusted CFG scale to {args.guidance_scale} for DiT.")

    main(args)
