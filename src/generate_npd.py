# src/generate_npd.py
import torch
import numpy as np
from PIL import Image
import os
import random
import argparse
from tqdm.auto import tqdm # Progress bar
import pandas as pd # For handling prompts
import time # For basic profiling
import traceback # For detailed error logs

# Diffusers imports
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

# HPSv2 import
try:
    import hpsv2
    HPSV2_AVAILABLE = True
except ImportError:
    print("Warning: hpsv2 library not found. Filtering based on HPSv2 score will be skipped.")
    print("Please install using: pip install hpsv2")
    HPSV2_AVAILABLE = False

# --- Configuration ---
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use torch.device object
DTYPE = torch.float16 # Use float16 for efficiency

# Re-denoise parameters (as per paper for SDXL)
CFG_L = 5.5 # Guidance scale for DDIM step
CFG_W = 1.0 # Guidance scale for DDIM-Inversion step

# Image generation parameters for evaluation (as per paper Appendix A.4)
EVAL_NUM_INFERENCE_STEPS = 10
EVAL_GUIDANCE_SCALE = 5.5 # Use standard CFG for generating images for HPSv2 eval
HPS_VERSION = "v2.1" # Or "v2.0", use the desired HPSv2 version

# --- Helper Functions ---

def set_seed(seed):
    """ Sets random seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_hps_score_wrapper(images, prompt, hps_version):
    """ Wrapper to call the hpsv2 library for scoring. """
    if not HPSV2_AVAILABLE:
        return torch.zeros(len(images) if isinstance(images, list) else 1)
    try:
        if isinstance(images, Image.Image): images = [images]
        valid_images = [img for img in images if img is not None]
        if not valid_images: return torch.zeros(len(images) if isinstance(images, list) else 1)
        scores_list = hpsv2.score(valid_images, prompt, hps_version=hps_version)
        final_scores = []
        score_idx = 0
        for img in images:
                if img is not None and score_idx < len(scores_list):
                    final_scores.append(scores_list[score_idx]); score_idx += 1
                else: final_scores.append(0.0)
        return torch.tensor(final_scores)
    except Exception as e:
        print(f"Error during HPSv2 scoring: {e}"); traceback.print_exc() # Print traceback for HPS errors
        return torch.zeros(len(images) if isinstance(images, list) else 1)

def predict_noise_cfg(pipe, scheduler, latents, text_embeddings_tuple, t, guidance_scale):
    """ Predicts noise using Classifier-Free Guidance for SDXL. """
    latent_model_input = torch.cat([latents] * 2)
    if hasattr(scheduler, "scale_model_input"):
            # Ensure t is tensor for scale_model_input if needed
            t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=latents.device)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t_tensor)

    text_embeddings, pooled_embeddings, add_time_ids = text_embeddings_tuple
    if add_time_ids is None: raise ValueError("add_time_ids cannot be None for SDXL UNet.")

    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, t, # Pass original t value
                                encoder_hidden_states=text_embeddings,
                                added_cond_kwargs={"text_embeds": pooled_embeddings, "time_ids": add_time_ids}
                                ).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred

def ddim_step(scheduler, noise_pred, t, latents):
    """ Performs one step of DDIM denoising using the scheduler's step function. """
    t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=latents.device, dtype=torch.long)
    if t_tensor.device != latents.device: t_tensor = t_tensor.to(latents.device)
    prev_latents = scheduler.step(noise_pred, t_tensor, latents).prev_sample
    return prev_latents

def ddim_inversion_step(scheduler, noise_pred, t, prev_latents):
    """ Performs one step of DDIM inversion (x_{t-1} -> x_t). """
    if isinstance(t, torch.Tensor): t_val = t.item()
    else: t_val = int(t)
    if not hasattr(scheduler, 'alphas_cumprod'): raise AttributeError("Scheduler lacks 'alphas_cumprod'.")

    alphas_cumprod = scheduler.alphas_cumprod.to(device=prev_latents.device, dtype=prev_latents.dtype)
    max_idx = len(alphas_cumprod) - 1
    t_idx = min(t_val, max_idx) # Clamp timestep value to max index
    t_idx = max(0, t_idx)       # Ensure non-negative index

    if not (0 <= t_idx <= max_idx): raise IndexError(f"Internal Error: Index {t_idx} still out of bounds.")

    alpha_prod_t = alphas_cumprod[t_idx]
    num_train_timesteps = scheduler.config.num_train_timesteps
    num_inference_steps = getattr(scheduler, 'num_inference_steps', num_train_timesteps)
    if num_inference_steps <= 0: num_inference_steps = num_train_timesteps
    step_size = num_train_timesteps // num_inference_steps if num_inference_steps > 0 else 1
    prev_timestep_idx = max(0, t_idx - step_size)
    if prev_timestep_idx > max_idx: prev_timestep_idx = max_idx
    alpha_prod_t_prev = alphas_cumprod[prev_timestep_idx] if prev_timestep_idx >= 0 else torch.tensor(1.0, device=alphas_cumprod.device, dtype=alphas_cumprod.dtype)

    alpha_prod_t_prev = torch.clamp(alpha_prod_t_prev, min=1e-6)
    ratio = alpha_prod_t / alpha_prod_t_prev
    ratio = torch.clamp(ratio, min=0.0)
    A = ratio ** 0.5
    term_inside_sqrt_B1 = alpha_prod_t * (1.0 - alpha_prod_t_prev) / alpha_prod_t_prev
    term_inside_sqrt_B1 = torch.clamp(term_inside_sqrt_B1, min=0.0)
    term1_B = term_inside_sqrt_B1 ** 0.5
    term_inside_sqrt_B2 = 1.0 - alpha_prod_t
    term_inside_sqrt_B2 = torch.clamp(term_inside_sqrt_B2, min=0.0)
    term2_B = term_inside_sqrt_B2 ** 0.5
    B = term1_B - term2_B
    inverted_latents = A * prev_latents - B * noise_pred
    return inverted_latents

# --- Main Data Generation Function ---
def generate_npd_data(prompts, pipe, scheduler, args):
    """ Generates the Noise Prompt Dataset. """
    selected_data = []
    processed_prompts = 0
    # Determine timesteps for re-denoise
    try:
        num_train_timesteps = scheduler.config.num_train_timesteps
        scheduler.set_timesteps(num_train_timesteps)
        if len(scheduler.timesteps) == 0: raise ValueError("Scheduler timesteps empty.")
        # Get the highest timestep VALUE (e.g., 1000 or 999)
        t_start_val = scheduler.timesteps[0]
        # Get the highest valid INDEX for alphas_cumprod (e.g., 999 if size is 1000)
        max_alphas_idx = len(scheduler.alphas_cumprod) - 1
        # Timestep value to pass to functions like predict_noise_cfg, step, inversion_step
        t_start_for_unet_and_step = t_start_val
        print(f"Debug: num_train_timesteps={num_train_timesteps}, t_start_val={t_start_val}, max_alphas_idx={max_alphas_idx}, len(alphas_cumprod)={len(scheduler.alphas_cumprod)}")
        if t_start_val > num_train_timesteps: print(f"Warning: t_start_val ({t_start_val}) > num_train_timesteps ({num_train_timesteps}).")
    except Exception as e: print(f"Error setting scheduler timesteps: {e}"); print(f"Scheduler config: {scheduler.config}"); return []

    eval_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    original_pipe_scheduler = pipe.scheduler
    pipe.scheduler = eval_scheduler

    pbar_total = args.max_samples if args.max_samples is not None else len(prompts)
    pbar = tqdm(total=pbar_total, desc="Selecting NPD Pairs")
    prompts_processed_count = 0

    for idx, prompt in enumerate(prompts):
        if args.max_samples is not None and len(selected_data) >= args.max_samples: break
        prompts_processed_count += 1
        if not isinstance(prompt, str) or not prompt: continue

        current_seed = args.start_seed + idx
        set_seed(current_seed)
        generator = torch.Generator(device=DEVICE).manual_seed(current_seed)
        start_time_prompt = time.time()

        # --- 1. Get Text Embeddings & Time IDs ---
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = None, None, None, None
        add_time_ids = None
        text_embeddings_for_unet = None
        try:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt=prompt, device=DEVICE, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt="",
            )
            if prompt_embeds is None or negative_prompt_embeds is None or pooled_prompt_embeds is None or negative_pooled_prompt_embeds is None:
                    raise ValueError("pipe.encode_prompt returned None for one or more embedding tensors.")

            orig_size = (args.output_size, args.output_size); target_size = (args.output_size, args.output_size); crops_coords_top_left = (0, 0)
            time_id_calculated = False
            # if hasattr(pipe, "_get_add_time_ids"):
            #     try:
            #         add_time_ids = pipe._get_add_time_ids(orig_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype).to(DEVICE)
            #         if add_time_ids is not None: time_id_calculated = True
            #     except Exception as e_int: print(f"Warning: pipe._get_add_time_ids failed: {e_int}")
            # if not time_id_calculated and hasattr(pipe, "get_add_time_ids"):
            #     try:
            #         add_time_ids = pipe.get_add_time_ids(orig_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype).to(DEVICE)
            #         if add_time_ids is not None: time_id_calculated = True
            #     except Exception as e_pub: print(f"Warning: pipe.get_add_time_ids failed: {e_pub}")
            # if not time_id_calculated:
            #     print(f"Warning: Using manual calculation for add_time_ids.")
            #     add_time_ids = torch.tensor([[orig_size[0], orig_size[1], crops_coords_top_left[0], crops_coords_top_left[1], target_size[0], target_size[1]]], dtype=prompt_embeds.dtype, device=DEVICE)
            #     if add_time_ids is not None: time_id_calculated = True
            add_time_ids = torch.tensor([[orig_size[0], orig_size[1], crops_coords_top_left[0], crops_coords_top_left[1], target_size[0], target_size[1]]], dtype=prompt_embeds.dtype, device=DEVICE)
            if add_time_ids is not None: time_id_calculated = True
            if add_time_ids is None: raise ValueError("Failed to calculate add_time_ids for SDXL.")

            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_embeddings = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            text_embeddings_for_unet = (text_embeddings, pooled_embeddings, add_time_ids)
        except Exception as e:
            print(f"Error during prompt encoding or time_id generation for '{prompt[:50]}...': {e}. Skipping.")
            traceback.print_exc() # Print full traceback for this error
            continue

        # --- 2. Generate Initial Noise ---
        latents_shape = (1, pipe.unet.config.in_channels, args.latent_height, args.latent_width)
        xt = randn_tensor(latents_shape, generator=generator, device=DEVICE, dtype=DTYPE)
        source_noise_cpu = xt.clone().cpu()

        # --- 3. Re-denoise Sampling ---
        reden_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        reden_scheduler.set_timesteps(num_train_timesteps) # Use full range for step values

        try:
            with torch.no_grad():
                # Use t_start_for_unet_and_step which holds the timestep VALUE
                epsilon_l = predict_noise_cfg(pipe, reden_scheduler, xt, text_embeddings_for_unet, t_start_for_unet_and_step, CFG_L)
                # xt_minus_1 = ddim_step(reden_scheduler, epsilon_l, t_start_for_unet_and_step, xt)
                xt_minus_1 = ddim_step(reden_scheduler, epsilon_l, max_alphas_idx, xt)
                epsilon_w = predict_noise_cfg(pipe, reden_scheduler, xt_minus_1, text_embeddings_for_unet, t_start_for_unet_and_step, CFG_W)
                # Set num_inference_steps for the scheduler before inversion
                reden_scheduler.num_inference_steps = num_train_timesteps # Ensure correct step calculation if needed
                xt_prime = ddim_inversion_step(reden_scheduler, epsilon_w, t_start_for_unet_and_step, xt_minus_1)
        except IndexError as e_idx:
            print(f"IndexError during re-denoise sampling for prompt '{prompt[:50]}...': {e_idx}. Check internal indexing.")
            traceback.print_exc()
            continue
        except Exception as e:
            print(f"Error during re-denoise sampling for prompt '{prompt[:50]}...': {e}. Skipping.")
            traceback.print_exc()
            continue

        target_noise_cpu = xt_prime.clone().cpu()

        # --- 4. AI Feedback Filtering ---
        if HPSV2_AVAILABLE:
            start_time_eval = time.time()
            pipe.scheduler = eval_scheduler
            pipe.scheduler.set_timesteps(EVAL_NUM_INFERENCE_STEPS)
# --- SIMPLIFIED common_eval_args for generation ---
            # Let the pipeline handle encoding and CFG internally
            common_eval_args_simplified = {
                "prompt": prompt, # Pass the prompt string directly
                "negative_prompt": "", # Provide empty negative prompt for CFG
                "height": args.output_size, # Still need height/width for SDXL time_ids calculation inside pipe
                "width": args.output_size,
                "num_inference_steps": EVAL_NUM_INFERENCE_STEPS,
                "guidance_scale": EVAL_GUIDANCE_SCALE, # CFG scale > 1 enables internal CFG
                "output_type": "pil"
                # REMOVED: prompt_embeds, pooled_*, negative_*, added_cond_kwargs
            }

            # Generate standard image
            set_seed(current_seed); generator = torch.Generator(device=DEVICE).manual_seed(current_seed)
            image_xt = None
            try:
                # Pass latents and generator separately
                with torch.no_grad():
                    image_xt = pipe(latents=xt, generator=generator, **common_eval_args_simplified).images[0]
            except Exception as e: print(f"Error generating standard image: {e}")

            # Generate golden image
            set_seed(current_seed); generator = torch.Generator(device=DEVICE).manual_seed(current_seed)
            image_xt_prime = None;
            try:
                # Pass latents and generator separately
                with torch.no_grad():
                    image_xt_prime = pipe(latents=xt_prime, generator=generator, **common_eval_args_simplified).images[0]
            except Exception as e: print(f"Error generating golden image: {e}")

            score_xt = calculate_hps_score_wrapper(image_xt, prompt, HPS_VERSION)[0] if image_xt else torch.tensor(0.0)
            score_xt_prime = calculate_hps_score_wrapper(image_xt_prime, prompt, HPS_VERSION)[0] if image_xt_prime else torch.tensor(0.0)
            eval_time = time.time() - start_time_eval

            if image_xt and image_xt_prime and score_xt_prime > score_xt:
                selected_data.append({
                    "prompt": prompt, "source_noise": source_noise_cpu, "target_noise": target_noise_cpu,
                    "score_source": score_xt.item(), "score_target": score_xt_prime.item(), "seed": current_seed,
                })
                pbar.update(1)
        else:
            selected_data.append({
                "prompt": prompt, "source_noise": source_noise_cpu, "target_noise": target_noise_cpu,
                "score_source": -1.0, "score_target": -1.0, "seed": current_seed,
            })
            pbar.n = len(selected_data); pbar.refresh()

        prompt_time = time.time() - start_time_prompt
        pbar.set_postfix({"Processed": f"{prompts_processed_count}/{len(prompts)}", "Selected": len(selected_data), "LastTime": f"{prompt_time:.2f}s"})
        if idx % 50 == 0: torch.cuda.empty_cache()

    pbar.close()
    pipe.scheduler = original_pipe_scheduler # Restore scheduler
    print(f"Finished processing {prompts_processed_count} prompts. Selected {len(selected_data)} pairs.")
    return selected_data

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Noise Prompt Dataset (NPD) using SDXL and HPSv2 filtering")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompts file (.txt or .csv).")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompts if CSV.")
    parser.add_argument("--output_dir", type=str, default="npd_dataset_sdxl", help="Directory to save the dataset.")
    parser.add_argument("--max_samples", type=int, default=None, help="Stop after selecting this many pairs. Default: process all.")
    parser.add_argument("--output_size", type=int, default=1024, help="Image size for SDXL.")
    parser.add_argument("--start_seed", type=int, default=42, help="Initial random seed offset.")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}"); print(f"Using dtype: {DTYPE}")
    if not HPSV2_AVAILABLE: print("WARNING: hpsv2 not installed. Data generated WITHOUT HPSv2 filtering.")

    set_seed(args.start_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    noise_dir = os.path.join(args.output_dir, "noises"); os.makedirs(noise_dir, exist_ok=True)

    print("Loading SDXL Pipeline...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(SDXL_MODEL_ID, torch_dtype=DTYPE, variant="fp16", use_safetensors=True, low_cpu_mem_usage=True).to(DEVICE)
        pipe.unet.eval(); pipe.vae.eval()
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None: pipe.text_encoder.eval()
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None: pipe.text_encoder_2.eval()
    except Exception as e: print(f"Error loading SDXL pipeline: {e}"); exit(1)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    print(f"Loading prompts from {args.prompt_file}...")
    prompts = []
    try:
        if args.prompt_file.lower().endswith(".csv"):
            df = pd.read_csv(args.prompt_file); prompts = df[args.prompt_column].astype(str).tolist()
        elif args.prompt_file.lower().endswith(".txt"):
            with open(args.prompt_file, 'r', encoding='utf-8') as f: prompts = [line.strip() for line in f if line.strip()]
        else: raise ValueError("Unsupported prompt file format.")
        prompts = [p for p in prompts if isinstance(p, str) and p]; print(f"Loaded {len(prompts)} valid prompts.")
        if not prompts: print("Error: No valid prompts found."); exit(1)
        if args.max_samples is not None: print(f"Will process until {args.max_samples} selected pairs are found.")
    except Exception as e: print(f"Error reading prompts: {e}"); exit(1)

    args.latent_height = args.output_size // pipe.vae_scale_factor
    args.latent_width = args.output_size // pipe.vae_scale_factor

    start_generation_time = time.time()
    selected_pairs = generate_npd_data(prompts, pipe, scheduler, args)
    end_generation_time = time.time(); print(f"Data generation took {end_generation_time - start_generation_time:.2f} seconds.")

    if selected_pairs:
        print("Saving selected data...")
        metadata = []; save_errors = 0
        for i, data in enumerate(tqdm(selected_pairs, desc="Saving files")):
            try:
                noise_filename_base = os.path.join(noise_dir, f"pair_{i:06d}")
                source_noise_path = f"{noise_filename_base}_source.pt"; target_noise_path = f"{noise_filename_base}_target.pt"
                torch.save(data["source_noise"].to(torch.float32), source_noise_path)
                torch.save(data["target_noise"].to(torch.float32), target_noise_path)
                metadata.append({ "id": i, "prompt": data["prompt"],
                    "source_noise_file": os.path.relpath(source_noise_path, args.output_dir),
                    "target_noise_file": os.path.relpath(target_noise_path, args.output_dir),
                    "score_source": data["score_source"], "score_target": data["score_target"], "seed": data["seed"] })
            except Exception as e: print(f"Error saving pair {i}: {e}"); save_errors += 1
        if save_errors > 0: print(f"Warning: Encountered {save_errors} errors while saving.")
        if metadata:
            metadata_df = pd.DataFrame(metadata); metadata_path = os.path.join(args.output_dir, "metadata.csv")
            try: metadata_df.to_csv(metadata_path, index=False); print(f"Saved metadata for {len(metadata)} pairs to {metadata_path}")
            except Exception as e: print(f"Error saving metadata CSV: {e}")
        else: print("No metadata saved.")
    else: print("No data pairs were selected or generated.")
    print("Dataset generation script finished.")
