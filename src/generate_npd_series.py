# src/generate_npd.py
import torch
import numpy as np
from PIL import Image # Keep PIL import in case needed elsewhere, though not for generation now
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

# --- Configuration ---
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use torch.device object
DTYPE = torch.float16 # Use float16 for efficiency

# Re-denoise parameters (as per paper for SDXL)
CFG_L = 5.5 # Guidance scale for DDIM step
CFG_W = 1.0 # Guidance scale for DDIM-Inversion step

# --- Helper Functions ---

def set_seed(seed):
    """ Sets random seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Removed calculate_hps_score_wrapper as filtering is disabled

def predict_noise_cfg(pipe, scheduler, latents, text_embeddings_tuple, t, guidance_scale):
    """ Predicts noise using Classifier-Free Guidance for SDXL. """
    latent_model_input = torch.cat([latents] * 2)
    if hasattr(scheduler, "scale_model_input"):
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=latents.device)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t_tensor)

    text_embeddings, pooled_embeddings, add_time_ids = text_embeddings_tuple
    if add_time_ids is None: raise ValueError("add_time_ids cannot be None for SDXL UNet.")

    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, t,
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
    t_idx = min(t_val, max_idx); t_idx = max(0, t_idx)
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
    ratio = alpha_prod_t / alpha_prod_t_prev; ratio = torch.clamp(ratio, min=0.0)
    A = ratio ** 0.5
    term_inside_sqrt_B1 = alpha_prod_t * (1.0 - alpha_prod_t_prev) / alpha_prod_t_prev
    term_inside_sqrt_B1 = torch.clamp(term_inside_sqrt_B1, min=0.0); term1_B = term_inside_sqrt_B1 ** 0.5
    term_inside_sqrt_B2 = 1.0 - alpha_prod_t
    term_inside_sqrt_B2 = torch.clamp(term_inside_sqrt_B2, min=0.0); term2_B = term_inside_sqrt_B2 ** 0.5
    B = term1_B - term2_B
    inverted_latents = A * prev_latents - B * noise_pred
    return inverted_latents

# --- Main Data Generation Function ---
def generate_npd_sequence_data(prompts, pipe, scheduler, args):
    """
    Generates Noise Prompt Dataset containing sequences of evolving golden noises.
    Does NOT perform HPSv2 filtering.
    """
    generated_data = [] # Store data for all processed prompts
    processed_prompts = 0

    # Determine timesteps for re-denoise
    try:
            num_train_timesteps = scheduler.config.num_train_timesteps
            scheduler.set_timesteps(num_train_timesteps)
            if len(scheduler.timesteps) == 0: raise ValueError("Scheduler timesteps empty.")
            t_start_val = scheduler.timesteps[0] # Highest timestep value
            max_alphas_idx = len(scheduler.alphas_cumprod) - 1 # Highest valid index
            t_start_for_unet_and_step = t_start_val # Value passed to funcs
            print(f"Debug: num_train_timesteps={num_train_timesteps}, t_start_val={t_start_val}, max_alphas_idx={max_alphas_idx}")
    except Exception as e: print(f"Error setting scheduler timesteps: {e}"); return []

    # Use a dedicated DDIM scheduler for the iterative process
    iter_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    iter_scheduler.set_timesteps(num_train_timesteps)

    pbar_total = args.max_prompts if args.max_prompts is not None else len(prompts)
    pbar = tqdm(total=pbar_total, desc="Generating Noise Sequences")

    for idx, prompt in enumerate(prompts):
        # Stop if max_prompts limit is reached
        if args.max_prompts is not None and processed_prompts >= args.max_prompts:
            break

        if not isinstance(prompt, str) or not prompt: continue

        current_seed = args.start_seed + idx
        set_seed(current_seed)
        generator = torch.Generator(device=DEVICE).manual_seed(current_seed)
        start_time_prompt = time.time()

        # --- 1. Get Text Embeddings & Time IDs ---
        text_embeddings_for_unet = None # Initialize
        try:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt=prompt, device=DEVICE, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt="",
            )
            if prompt_embeds is None: raise ValueError("encode_prompt failed.") # Basic check
            orig_size = (args.output_size, args.output_size); target_size = (args.output_size, args.output_size); crops_coords_top_left = (0, 0)
            add_time_ids = torch.tensor([[orig_size[0], orig_size[1], crops_coords_top_left[0], crops_coords_top_left[1], target_size[0], target_size[1]]], dtype=prompt_embeds.dtype, device=DEVICE)
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_embeddings = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            text_embeddings_for_unet = (text_embeddings, pooled_embeddings, add_time_ids)
        except Exception as e:
            print(f"Error encoding prompt '{prompt[:50]}...': {e}. Skipping.")
            traceback.print_exc()
            continue # Skip this prompt

        # --- 2. Generate Initial Noise ---
        latents_shape = (1, pipe.unet.config.in_channels, args.latent_height, args.latent_width)
        xt = randn_tensor(latents_shape, generator=generator, device=DEVICE, dtype=DTYPE)
        source_noise_cpu = xt.clone().cpu()

        # --- 3. Iterative Re-denoise Sampling ---
        golden_noise_sequence = [] # List to store the sequence x'_1, x'_2, ...
        current_xt = xt # Start iteration from the initial noise
        iteration_successful = True

        for k in range(args.num_steps): # Loop n times
            try:
                with torch.no_grad():
                    # Apply DDIM Denoise + DDIM Inversion
                    epsilon_l = predict_noise_cfg(pipe, iter_scheduler, current_xt, text_embeddings_for_unet, t_start_for_unet_and_step, CFG_L)
                    # Use max_alphas_idx for the ddim_step call
                    xt_minus_1 = ddim_step(iter_scheduler, epsilon_l, max_alphas_idx, current_xt)
                    epsilon_w = predict_noise_cfg(pipe, iter_scheduler, xt_minus_1, text_embeddings_for_unet, t_start_for_unet_and_step, CFG_W)
                    # Set num_inference_steps for the scheduler before inversion
                    iter_scheduler.num_inference_steps = num_train_timesteps
                    # Get the next noise in the sequence
                    next_xt = ddim_inversion_step(iter_scheduler, epsilon_w, t_start_for_unet_and_step, xt_minus_1)

                # Store the result (on CPU to save GPU memory)
                golden_noise_sequence.append(next_xt.clone().cpu())
                # Update current_xt for the next iteration
                current_xt = next_xt

            except Exception as e:
                print(f"Error during re-denoise step {k+1} for prompt '{prompt[:50]}...': {e}. Stopping sequence generation for this prompt.")
                traceback.print_exc()
                iteration_successful = False
                break # Stop generating sequence for this prompt if an error occurs

        # --- 4. Store Data if sequence generation was successful ---
        if iteration_successful and len(golden_noise_sequence) == args.num_steps:
            generated_data.append({
                "prompt": prompt,
                "source_noise": source_noise_cpu,
                "golden_sequence": golden_noise_sequence, # Store the list of tensors
                "seed": current_seed,
                "num_steps": args.num_steps
            })
            pbar.update(1) # Update progress bar after successfully processing one prompt
        else:
                print(f"Skipping prompt '{prompt[:50]}...' due to errors during sequence generation.")


        processed_prompts += 1
        prompt_time = time.time() - start_time_prompt
        pbar.set_postfix({"Processed": f"{processed_prompts}/{len(prompts)}", "Generated": len(generated_data), "LastTime": f"{prompt_time:.2f}s"})

        if idx % 50 == 0: torch.cuda.empty_cache()

    pbar.close()
    print(f"Finished processing {processed_prompts} prompts. Generated {len(generated_data)} sequences.")
    return generated_data

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Noise Prompt Dataset (NPD) with noise sequences using SDXL")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompts file (.txt or .csv).")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompts if CSV.")
    parser.add_argument("--output_dir", type=str, default="npd_sequence_dataset_sdxl", help="Directory to save the dataset.")
    # Changed max_samples to max_prompts for clarity
    parser.add_argument("--max_prompts", type=int, default=None, help="Maximum number of prompts to process. Default: process all.")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of iterative re-denoise steps to generate per prompt.")
    parser.add_argument("--output_size", type=int, default=1024, help="Image size for SDXL.")
    parser.add_argument("--start_seed", type=int, default=42, help="Initial random seed offset.")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}"); print(f"Using dtype: {DTYPE}")

    set_seed(args.start_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    noise_dir = os.path.join(args.output_dir, "sequences") # Changed directory name
    os.makedirs(noise_dir, exist_ok=True)

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
        if args.max_prompts is not None: print(f"Will process a maximum of {args.max_prompts} prompts.")
    except Exception as e: print(f"Error reading prompts: {e}"); exit(1)

    args.latent_height = args.output_size // pipe.vae_scale_factor
    args.latent_width = args.output_size // pipe.vae_scale_factor

    start_generation_time = time.time()
    # Call the modified generation function
    generated_sequences = generate_npd_sequence_data(prompts, pipe, scheduler, args)
    end_generation_time = time.time(); print(f"Data generation took {end_generation_time - start_generation_time:.2f} seconds.")

    if generated_sequences:
        print("Saving generated data...")
        metadata = []; save_errors = 0
        for i, data in enumerate(tqdm(generated_sequences, desc="Saving files")):
            try:
                # Save source noise
                source_filename = os.path.join(noise_dir, f"pair_{i:06d}_source.pt")
                torch.save(data["source_noise"].to(torch.float32), source_filename)

                # Save golden sequence (list of tensors)
                sequence_filename = os.path.join(noise_dir, f"pair_{i:06d}_golden_sequence.pt")
                # Ensure all tensors in the list are float32 before saving
                sequence_to_save = [t.to(torch.float32) for t in data["golden_sequence"]]
                torch.save(sequence_to_save, sequence_filename)

                metadata.append({
                    "id": i, "prompt": data["prompt"],
                    "source_noise_file": os.path.relpath(source_filename, args.output_dir),
                    "golden_sequence_file": os.path.relpath(sequence_filename, args.output_dir),
                    "num_steps": data["num_steps"], # Store the length of the sequence
                    "seed": data["seed"]
                })
            except Exception as e: print(f"Error saving sequence {i}: {e}"); save_errors += 1
        if save_errors > 0: print(f"Warning: Encountered {save_errors} errors while saving files.")
        if metadata:
            metadata_df = pd.DataFrame(metadata); metadata_path = os.path.join(args.output_dir, "metadata.csv")
            try: metadata_df.to_csv(metadata_path, index=False); print(f"Saved metadata for {len(metadata)} sequences to {metadata_path}")
            except Exception as e: print(f"Error saving metadata CSV: {e}")
        else: print("No metadata saved.")
    else: print("No data sequences were generated or saved.")
    print("Dataset generation script finished.")

