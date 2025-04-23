import torch
import numpy as np
from PIL import Image
import os
import random
import argparse
from tqdm.auto import tqdm # Progress bar
import pandas as pd # For handling prompts
import time # For basic profiling

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
        # Return dummy scores if hpsv2 is not installed
        return torch.zeros(len(images) if isinstance(images, list) else 1)

    try:
        # hpsv2.score expects a list of images or a single image
        # and a single prompt string.
        # Ensure images is a list even if it's a single PIL image
        if isinstance(images, Image.Image):
            images = [images]

        # The hpsv2 library returns a list of scores
        scores = hpsv2.score(images, prompt, hps_version=hps_version)
        return torch.tensor(scores) # Convert list of scores to a tensor
    except Exception as e:
        print(f"Error during HPSv2 scoring: {e}")
        # Return dummy scores in case of error
        return torch.zeros(len(images) if isinstance(images, list) else 1)

def predict_noise_cfg(pipe, scheduler, latents, text_embeddings_tuple, t, guidance_scale):
    """ Predicts noise using Classifier-Free Guidance for SDXL. """
    # Expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # Unpack text embeddings tuple
    text_embeddings, pooled_embeddings, add_time_ids = text_embeddings_tuple

    # Predict the noise residual
    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings, # Shape (2, seq_len, dim)
            added_cond_kwargs={"text_embeds": pooled_embeddings, "time_ids": add_time_ids} # SDXL specific pooled embeds and time_ids (Shape: 2, pooled_dim)
        ).sample

    # Perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred

def ddim_step(scheduler, noise_pred, t, latents):
    """ Performs one step of DDIM denoising using the scheduler's step function. """
    # compute the previous noisy sample x_t -> x_{t-1}
    # Ensure t is a tensor
    if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=latents.device)
    prev_latents = scheduler.step(noise_pred, t, latents).prev_sample
    return prev_latents

def ddim_inversion_step(scheduler, noise_pred, t, prev_latents):
    """
    Performs one step of DDIM inversion (x_{t-1} -> x_t).
    This implementation uses a derived formula based on rearranging the DDIM step equation.
    WARNING: This formula can be sensitive and requires careful validation.
                It approximates epsilon(x_t, t) with epsilon(x_{t-1}, t).
    """
    # Ensure t is an integer index for accessing alphas_cumprod
    if isinstance(t, torch.Tensor):
        t_idx = t.item()
    else:
        t_idx = int(t) # Make sure t is usable as an index

    # Get alpha products
    alpha_prod_t = scheduler.alphas_cumprod[t_idx]
    # Calculate approximate previous timestep index for alpha_prod_t_prev
    # This assumes uniform spacing, which might not be true for all schedulers/settings
    prev_timestep_idx = max(0, t_idx - scheduler.config.num_train_timesteps // scheduler.num_inference_steps)
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep_idx] if prev_timestep_idx >= 0 else scheduler.final_alpha_cumprod

    # Calculate the coefficient based on the derived formula:
    # x_t = A * x_{t-1} - B * epsilon
    # A = (alpha_prod_t / alpha_prod_t_prev) ** 0.5
    # B = [ (alpha_prod_t * (1 - alpha_prod_t_prev) / alpha_prod_t_prev) ** 0.5 - (1 - alpha_prod_t) ** 0.5 ]

    # Handle potential division by zero or negative values under sqrt
    if alpha_prod_t_prev == 0: alpha_prod_t_prev = 1e-6 # Avoid division by zero
    ratio = alpha_prod_t / alpha_prod_t_prev
    if ratio < 0: ratio = 0 # Avoid sqrt of negative

    A = ratio ** 0.5

    term_inside_sqrt = alpha_prod_t * (1 - alpha_prod_t_prev) / alpha_prod_t_prev
    if term_inside_sqrt < 0: term_inside_sqrt = 0 # Avoid sqrt of negative
    term1_B = term_inside_sqrt ** 0.5

    term2_B_inside_sqrt = 1 - alpha_prod_t
    if term2_B_inside_sqrt < 0: term2_B_inside_sqrt = 0 # Avoid sqrt of negative
    term2_B = term2_B_inside_sqrt ** 0.5

    B = term1_B - term2_B

    # Calculate inverted latents
    inverted_latents = A * prev_latents - B * noise_pred

    return inverted_latents


# --- Main Data Generation Function ---
def generate_npd_data(prompts, pipe, scheduler, args):
    """ Generates the Noise Prompt Dataset. """

    selected_data = []
    processed_prompts = 0
    total_prompts_to_process = len(prompts) if args.max_samples is None else min(len(prompts), args.max_samples * 5) # Estimate more to account for filtering

    # Determine timesteps for re-denoise (needs full scheduler steps)
    scheduler.set_timesteps(args.num_train_timesteps)
    if len(scheduler.timesteps) == 0:
            print("Error: Scheduler timesteps not initialized correctly.")
            return []
    t_start = scheduler.timesteps[0] # Typically T-1 (e.g., 999)

    # Set scheduler for eval generation (using fewer steps)
    eval_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = eval_scheduler # Temporarily set eval scheduler in pipe

    pbar = tqdm(total=args.max_samples if args.max_samples is not None else len(prompts), desc="Selecting NPD Pairs")

    for idx, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt:
            # print(f"Skipping invalid prompt at index {idx}: {prompt}")
            continue

        # Set seed for this specific prompt pair generation
        # Use a combination of start_seed and index for more deterministic runs if needed
        current_seed = args.start_seed + idx
        set_seed(current_seed)
        generator = torch.Generator(device=DEVICE).manual_seed(current_seed)

        start_time_prompt = time.time()

        # --- 1. Get Text Embeddings (Conditional and Unconditional) ---
        try:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                device=DEVICE,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
            # Combine for CFG usage
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_embeddings = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

            # Get add_time_ids needed for SDXL UNet
            bsz = 1
            orig_size = (args.output_size, args.output_size)
            target_size = (args.output_size, args.output_size)
            # Ensure pipe has access to _get_add_time_ids (might be protected)
            if hasattr(pipe, "_get_add_time_ids"):
                    add_time_ids = pipe._get_add_time_ids(orig_size, (0,0), target_size, dtype=prompt_embeds.dtype).to(DEVICE)
            else: # Fallback if method is not accessible
                    print("Warning: Cannot access pipe._get_add_time_ids. Using placeholder time_ids.")
                    # Placeholder values - check SDXL pipeline internals for correct calculation
                    add_time_ids = torch.tensor([[orig_size[0], orig_size[1], 0, 0, target_size[0], target_size[1]]], dtype=prompt_embeds.dtype, device=DEVICE)

            add_time_ids = torch.cat([add_time_ids] * 2) # Repeat for CFG

            text_embeddings_for_unet = (text_embeddings, pooled_embeddings, add_time_ids)

        except Exception as e:
            print(f"Error encoding prompt '{prompt[:50]}...': {e}. Skipping.")
            continue

        # --- 2. Generate Initial Noise (Source Noise) ---
        latents_shape = (1, pipe.unet.config.in_channels, args.latent_height, args.latent_width)
        xt = randn_tensor(latents_shape, generator=generator, device=DEVICE, dtype=DTYPE)
        source_noise_cpu = xt.clone().cpu() # Store original noise on CPU

        # --- 3. Re-denoise Sampling to get Target Noise ---
        # Use the DDIM scheduler for these steps
        reden_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        reden_scheduler.set_timesteps(args.num_train_timesteps) # Ensure full timesteps available
            # Ensure t_start is a valid index
        if t_start >= len(reden_scheduler.alphas_cumprod):
                print(f"Error: t_start {t_start} is out of bounds for scheduler alphas_cumprod (size {len(reden_scheduler.alphas_cumprod)}). Skipping prompt.")
                continue

        try:
            with torch.no_grad():
                # Predict noise for DDIM step (using xt, timestep t_start, CFG_L)
                epsilon_l = predict_noise_cfg(pipe, reden_scheduler, xt, text_embeddings_for_unet, t_start, CFG_L)
                # Perform DDIM step: xt -> x_{t-1}
                xt_minus_1 = ddim_step(reden_scheduler, epsilon_l, t_start, xt)

                # Predict noise for DDIM-Inversion step (using xt_minus_1, timestep t_start, CFG_W)
                epsilon_w = predict_noise_cfg(pipe, reden_scheduler, xt_minus_1, text_embeddings_for_unet, t_start, CFG_W)
                # Perform DDIM-Inversion step: x_{t-1} -> x'_t
                # Set the number of inference steps for the scheduler before inversion
                reden_scheduler.num_inference_steps = args.num_train_timesteps # Ensure correct step calculation
                xt_prime = ddim_inversion_step(reden_scheduler, epsilon_w, t_start, xt_minus_1)

        except Exception as e:
            print(f"Error during re-denoise sampling for prompt '{prompt[:50]}...': {e}. Skipping.")
            continue

        target_noise_cpu = xt_prime.clone().cpu() # Store potential golden noise on CPU

        # --- 4. AI Feedback Filtering ---
        if HPSV2_AVAILABLE:
            start_time_eval = time.time()
            # Generate images for evaluation using standard pipeline settings
            common_eval_args = {
                # Pass embeds directly for consistency and efficiency
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                "num_inference_steps": EVAL_NUM_INFERENCE_STEPS,
                "guidance_scale": EVAL_GUIDANCE_SCALE,
                "output_type": "pil"
            }

            # Reset generator for fair comparison
            set_seed(current_seed)
            generator = torch.Generator(device=DEVICE).manual_seed(current_seed)
            # Generate image from source noise
            try:
                with torch.no_grad():
                        image_xt = pipe(latents=xt, generator=generator, **common_eval_args).images[0]
            except Exception as e:
                    print(f"Error generating image from source noise for prompt '{prompt[:50]}...': {e}. Skipping filtering.")
                    score_xt = torch.tensor(-1.0) # Indicate error
                    image_xt = None


            # Reset generator again
            set_seed(current_seed)
            generator = torch.Generator(device=DEVICE).manual_seed(current_seed)
            # Generate image from target noise
            try:
                with torch.no_grad():
                        image_xt_prime = pipe(latents=xt_prime, generator=generator, **common_eval_args).images[0]
            except Exception as e:
                    print(f"Error generating image from target noise for prompt '{prompt[:50]}...': {e}. Skipping filtering.")
                    score_xt_prime = torch.tensor(-1.0) # Indicate error
                    image_xt_prime = None

            # Calculate HPSv2 scores only if images were generated successfully
            score_xt = calculate_hps_score_wrapper(image_xt, prompt, HPS_VERSION)[0] if image_xt else torch.tensor(-1.0)
            score_xt_prime = calculate_hps_score_wrapper(image_xt_prime, prompt, HPS_VERSION)[0] if image_xt_prime else torch.tensor(-1.0)

            eval_time = time.time() - start_time_eval

            # Filter based on score (m=0 threshold) and successful generation
            if score_xt_prime > score_xt and score_xt.item() != -1.0 and score_xt_prime.item() != -1.0 :
                selected_data.append({
                    "prompt": prompt,
                    "source_noise": source_noise_cpu,
                    "target_noise": target_noise_cpu,
                    "score_source": score_xt.item(),
                    "score_target": score_xt_prime.item(),
                    "seed": current_seed,
                })
                pbar.update(1) # Update progress bar only when an item is selected
                # print(f"Selected pair for prompt: {prompt[:50]}... Scores: {score_xt.item():.4f} -> {score_xt_prime.item():.4f} (Eval time: {eval_time:.2f}s)")
            # else:
                # Optional: Log discarded pairs
                # print(f"Discarded pair for prompt: {prompt[:50]}... Scores: {score_xt.item():.4f} -> {score_xt_prime.item():.4f} (Eval time: {eval_time:.2f}s)")
        else:
                # If HPSv2 is not available, save all generated pairs without filtering
                selected_data.append({
                    "prompt": prompt,
                    "source_noise": source_noise_cpu,
                    "target_noise": target_noise_cpu,
                    "score_source": -1.0, # Indicate no score
                    "score_target": -1.0, # Indicate no score
                    "seed": current_seed,
                })
                pbar.update(1)


        processed_prompts += 1
        prompt_time = time.time() - start_time_prompt
        pbar.set_postfix({"Processed": f"{processed_prompts}/{len(prompts)}", "LastPromptTime": f"{prompt_time:.2f}s"})

        # Check if max_samples limit is reached
        if args.max_samples is not None and len(selected_data) >= args.max_samples:
            print(f"\nReached max samples ({args.max_samples}). Stopping.")
            break

        # Optional: Clear CUDA cache periodically if memory becomes an issue
        if idx % 50 == 0:
                torch.cuda.empty_cache()

    pbar.close()
    print(f"Finished processing. Selected {len(selected_data)} pairs out of {processed_prompts} prompts processed.")
    return selected_data

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Noise Prompt Dataset (NPD) using SDXL and HPSv2 filtering")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to a file containing prompts (e.g., txt or csv). Assumes one prompt per line if txt, or a 'prompt' column if csv.")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name containing prompts if using a CSV file.")
    parser.add_argument("--output_dir", type=str, default="npd_dataset_sdxl", help="Directory to save the dataset.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of noise pairs to generate and save. Default: process all prompts.")
    parser.add_argument("--output_size", type=int, default=1024, help="Image size for SDXL.")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of training timesteps for the scheduler (used for indexing).")
    parser.add_argument("--start_seed", type=int, default=42, help="Initial random seed offset.")
    # Batching is complex to implement correctly here, keeping it at 1.
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing prompts (Currently only supports 1).")

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Using dtype: {DTYPE}")
    if not HPSV2_AVAILABLE:
        print("WARNING: hpsv2 library not found. Data will be generated WITHOUT HPSv2 filtering.")

    set_seed(args.start_seed) # Set initial seed for reproducibility of the run itself
    os.makedirs(args.output_dir, exist_ok=True)
    noise_dir = os.path.join(args.output_dir, "noises")
    os.makedirs(noise_dir, exist_ok=True)

    # --- Load Models ---
    print("Loading SDXL Pipeline...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_ID, torch_dtype=DTYPE, variant="fp16", use_safetensors=True
        ).to(DEVICE)
        pipe.unet.eval() # Ensure UNet is in eval mode
        pipe.vae.eval() # Ensure VAE is in eval mode
        # pipe.text_encoder.eval() # Ensure text encoders are in eval mode
        # pipe.text_encoder_2.eval()
    except Exception as e:
            print(f"Error loading SDXL pipeline: {e}")
            print("Please ensure you have enough VRAM and the model ID is correct.")
            exit(1)

    # Use DDIM Scheduler for the re-denoise steps
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # --- Load Prompts ---
    print(f"Loading prompts from {args.prompt_file}...")
    prompts = []
    try:
        if args.prompt_file.lower().endswith(".csv"):
            df = pd.read_csv(args.prompt_file)
            if args.prompt_column not in df.columns:
                raise ValueError(f"Column '{args.prompt_column}' not found in CSV file.")
            prompts = df[args.prompt_column].astype(str).tolist()
        elif args.prompt_file.lower().endswith(".txt"):
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
                raise ValueError("Unsupported prompt file format. Use .txt or .csv")

        prompts = [p for p in prompts if isinstance(p, str) and p] # Filter out empty/invalid prompts
        print(f"Loaded {len(prompts)} valid prompts.")
        if not prompts:
                print("Error: No valid prompts found in the file.")
                exit(1)

        if args.max_samples is not None:
                print(f"Will process prompts until {args.max_samples} selected pairs are found.")

    except FileNotFoundError:
        print(f"Error: Prompt file not found at {args.prompt_file}")
        exit(1)
    except Exception as e:
        print(f"Error reading prompts: {e}")
        exit(1)

    # Calculate latent dimensions
    args.latent_height = args.output_size // pipe.vae_scale_factor
    args.latent_width = args.output_size // pipe.vae_scale_factor

    # --- Generate Data ---
    start_generation_time = time.time()
    selected_pairs = generate_npd_data(prompts, pipe, scheduler, args)
    end_generation_time = time.time()
    print(f"Data generation took {end_generation_time - start_generation_time:.2f} seconds.")


    # --- Save Data ---
    if selected_pairs:
        print("Saving selected data...")
        metadata = []
        save_errors = 0
        for i, data in enumerate(tqdm(selected_pairs, desc="Saving files")):
            try:
                noise_filename_base = os.path.join(noise_dir, f"pair_{i:06d}")
                source_noise_path = f"{noise_filename_base}_source.pt"
                target_noise_path = f"{noise_filename_base}_target.pt"

                # Ensure tensors are float32 for potentially wider compatibility, though float16 saves space
                torch.save(data["source_noise"].to(torch.float32), source_noise_path)
                torch.save(data["target_noise"].to(torch.float32), target_noise_path)

                metadata.append({
                    "id": i,
                    "prompt": data["prompt"],
                    "source_noise_file": os.path.relpath(source_noise_path, args.output_dir), # Relative path
                    "target_noise_file": os.path.relpath(target_noise_path, args.output_dir), # Relative path
                    "score_source": data["score_source"],
                    "score_target": data["score_target"],
                    "seed": data["seed"]
                })
            except Exception as e:
                print(f"Error saving pair {i}: {e}")
                save_errors += 1

        if save_errors > 0:
                print(f"Warning: Encountered {save_errors} errors while saving noise files.")

        if metadata:
            metadata_df = pd.DataFrame(metadata)
            metadata_path = os.path.join(args.output_dir, "metadata.csv")
            try:
                metadata_df.to_csv(metadata_path, index=False)
                print(f"Saved metadata for {len(metadata)} pairs to {metadata_path}")
            except Exception as e:
                print(f"Error saving metadata CSV: {e}")
        else:
                print("No metadata saved as no pairs were successfully processed and saved.")

    else:
        print("No data pairs were selected or generated.")

    print("Dataset generation script finished.")