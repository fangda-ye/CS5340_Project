# src/train_seq_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import os
import time
from accelerate import Accelerator, DistributedDataParallelKwargs
import math
import traceback

# Import sequence model and dataset loader
try:
    from model import NoiseSequenceTransformer # Import the sequence transformer
    # Use the updated dataset loader function
    from dataset import get_sequence_dataloader
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure model and dataset modules are in the correct path.")
    exit(1)

# Base pipeline import for text encoding
from diffusers import StableDiffusionXLPipeline, HunyuanDiTPipeline # Add others if needed

# --- Loss Function ---
def gaussian_nll_loss(mean_pred, log_var_pred, target, reduction='mean'):
    """ Calculates the Gaussian Negative Log-Likelihood loss element-wise. """
    # Ensure variance is stable
    log_var_pred = torch.clamp(log_var_pred, min=-10, max=10)
    inv_var = torch.exp(-log_var_pred) # 1 / sigma^2
    # Calculate MSE term weighted by inverse variance
    mse_term = (target - mean_pred)**2 * inv_var
    # Combine terms: 0.5 * ( (x-mu)^2 / sigma^2 + log(sigma^2) ) + const
    # We drop the constant log(2*pi)
    loss = 0.5 * (mse_term + log_var_pred)

    if reduction == 'mean':
        return loss.mean() # Average over all elements (batch, seq, c, h, w)
    elif reduction == 'sum':
        return loss.sum()
    else: # 'none'
        return loss

# --- Main Training Function ---
def main(args):
    # --- Initialize Accelerator ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) # Set to True if model has unused params
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device
    if accelerator.is_main_process:
            accelerator.init_trackers("seq_model_training")

    accelerator.print(f"Using device: {device}, Mixed precision: {args.mixed_precision}")
    accelerator.print(f"Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # --- Load Base Model Pipeline (for text encoding) ---
    # (Same logic as train.py to load appropriate text encoder based on args.base_model_id)
    accelerator.print(f"Loading base model pipeline ({args.base_model_id}) for text encoder...")
    base_dtype = torch.float16 if args.mixed_precision != 'no' else torch.float32
    text_encoder, text_encoder_2 = None, None # Initialize
    try:
        if "xl" in args.base_model_id.lower() and "hunyuan" not in args.base_model_id.lower():
            pipe_class = StableDiffusionXLPipeline
            base_pipeline = pipe_class.from_pretrained(args.base_model_id, torch_dtype=base_dtype, variant="fp16", use_safetensors=True, low_cpu_mem_usage=True)
            text_encoder = base_pipeline.text_encoder
            text_encoder_2 = base_pipeline.text_encoder_2
            # Store tokenizer if needed (usually not for just encoding)
            # tokenizer = base_pipeline.tokenizer
            # tokenizer_2 = base_pipeline.tokenizer_2
            # We need the final embedding function
            def encode_text(prompt_list):
                    with torch.no_grad():
                        prompt_embeds, _, pooled_prompt_embeds, _ = base_pipeline.encode_prompt(
                            prompt=prompt_list, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
                        )
                        # Decide which embedding to use for the seq model (e.g., prompt_embeds or pooled)
                        # Let's assume we use prompt_embeds [B, SeqLen, Dim]
                        return prompt_embeds
            del base_pipeline.vae, base_pipeline.unet # Free memory

        elif "hunyuan" in args.base_model_id.lower():
                pipe_class = HunyuanDiTPipeline
                base_pipeline = pipe_class.from_pretrained(args.base_model_id, torch_dtype=base_dtype, low_cpu_mem_usage=True)
                text_encoder = getattr(base_pipeline, "text_encoder", None)
                # tokenizer = getattr(base_pipeline, "tokenizer", None)
                def encode_text(prompt_list):
                    with torch.no_grad():
                        # Check how Hunyuan encodes prompts - might return different structure
                        embed_output = base_pipeline.encode_prompt(prompt=prompt_list, device=device)
                        # Assume first element is the sequence embedding
                        return embed_output[0] if isinstance(embed_output, tuple) else embed_output
                del base_pipeline.transformer, base_pipeline.vae # Adjust names
                text_encoder_2 = None
        else:
            raise NotImplementedError(f"Base pipeline loading not implemented for {args.base_model_id}")

        if text_encoder: text_encoder.to(device).requires_grad_(False).eval()
        if text_encoder_2: text_encoder_2.to(device).requires_grad_(False).eval()

    except Exception as e:
            accelerator.print(f"Error loading base pipeline for text encoding: {e}"); exit(1)
    accelerator.print("Base model text encoder(s) loaded.")
    torch.cuda.empty_cache()


    # --- Initialize Noise Sequence Transformer Model ---
    accelerator.print("Initializing NoiseSequenceTransformer model...")
    # Determine text_embed_dim based on loaded encoder(s)
    # This needs careful checking based on the actual output of encode_text
    # Example: For SDXL using prompt_embeds, it might be text_encoder's dim (e.g., 768 or 1024)
    # Let's assume a placeholder or make it an argument
    # text_embed_dim = text_encoder.config.hidden_size if text_encoder else 768 # Example placeholder
    if args.npnet_model_id == 'DiT':
            text_embed_dim = 1024 # Placeholder based on DiT context
    else: # SDXL / DreamShaper
            text_embed_dim = 1024 # Placeholder for CLIP-L, adjust if using CLIP-G (1280) or combined

    model = NoiseSequenceTransformer(
        text_embed_dim=args.text_embed_dim, # Pass dimension as argument
        transformer_d_model=args.d_model,
        transformer_nhead=args.nhead,
        transformer_num_encoder_layers=args.num_encoder_layers,
        transformer_num_decoder_layers=args.num_decoder_layers,
        transformer_dim_feedforward=args.dim_feedforward,
        transformer_dropout=args.dropout,
        noise_img_size=args.noise_resolution,
        noise_patch_size=args.patch_size,
        noise_in_chans=args.noise_channels,
        predict_variance=args.predict_variance,
        max_seq_len=args.max_seq_len # Pass max sequence length
    )

    # Load pretrained weights if specified
    if args.resume_from:
        accelerator.print(f"Resuming training from checkpoint: {args.resume_from}")
        try:
                model.load_state_dict(torch.load(args.resume_from, map_location="cpu"))
                accelerator.print("Resumed weights loaded successfully.")
        except Exception as e:
                accelerator.print(f"Error loading resume checkpoint: {e}. Starting from scratch.")


    # --- Prepare Dataset and DataLoader ---
    accelerator.print("Loading sequence dataset...")
    metadata_path = os.path.join(args.dataset_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
            accelerator.print(f"Error: metadata.csv not found in {args.dataset_dir}"); exit(1)
    train_dataloader = get_sequence_dataloader(
        metadata_file=metadata_path,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len -1, # Load N steps for target, need N-1 + source for input
        shuffle=True,
        num_workers=args.num_workers
    )
    if train_dataloader is None: accelerator.print("Failed to create DataLoader."); exit(1)

    # --- Optimizer and Loss ---
    accelerator.print("Setting up optimizer and loss...")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    # Use NLL loss if predicting variance, otherwise MSE
    if args.predict_variance:
        criterion = gaussian_nll_loss
        print("Using Gaussian NLL Loss.")
    else:
        criterion = nn.MSELoss()
        print("Using MSE Loss.")

    # --- Prepare with Accelerator ---
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # --- Training Loop ---
    accelerator.print("Starting training...")
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * args.num_epochs
    if args.max_train_steps: total_steps = min(args.max_train_steps, total_steps)
    accelerator.print(f"Total optimization steps: {total_steps}")

    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, desc="Training Steps")
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        epoch_start_time = time.time()

        for step, batch in enumerate(train_dataloader):
                if batch is None: accelerator.print(f"Warning: Skipping empty batch at step {step}."); continue

                with accelerator.accumulate(model):
                # --- Get Data ---
                prompts = batch['prompt']
                source_noise = batch['source_noise'] # [B, C, H, W]
                golden_sequence = batch['golden_sequence'] # [B, SeqLen, C, H, W]
                B, T_target, C, H, W = golden_sequence.shape

                # --- Prepare Model Inputs ---
                # Input sequence: [x_T, x'_1, ..., x'_{N-1}]
                # Target sequence: [x'_1, ..., x'_N]
                # Need source noise with sequence dimension: [B, 1, C, H, W]
                source_noise_seq = source_noise.unsqueeze(1)
                # Input sequence for transformer (teacher forcing)
                input_noise_seq = torch.cat([source_noise_seq, golden_sequence[:, :-1]], dim=1) # [B, T_target, C, H, W]
                target_noise_seq = golden_sequence # [B, T_target, C, H, W]

                # --- Encode Prompts ---
                try:
                    with torch.no_grad():
                            text_embed = encode_text(prompts) # [B, SeqLen_txt, D_txt] or [B, D_txt]
                            # Ensure text_embed is on the correct device
                            text_embed = text_embed.to(device)
                except Exception as e:
                    accelerator.print(f"Error encoding prompts at step {step}: {e}. Skipping batch."); continue

                # --- Forward Pass ---
                # Ensure inputs are float32 if model expects it
                input_noise_seq_f32 = input_noise_seq.to(torch.float32)
                text_embed_f32 = text_embed.to(torch.float32)
                target_noise_seq_f32 = target_noise_seq.to(torch.float32)

                if args.predict_variance:
                    mean_pred_seq, log_var_pred_seq = model(input_noise_seq_f32, text_embed_f32)
                    loss = criterion(mean_pred_seq, log_var_pred_seq, target_noise_seq_f32)
                else:
                    mean_pred_seq = model(input_noise_seq_f32, text_embed_f32)
                    loss = criterion(mean_pred_seq, target_noise_seq_f32)

                # Accumulate loss
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # --- Backward Pass & Optimize ---
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    global_step += 1

                    # --- Logging ---
                    if global_step % args.log_steps == 0:
                        logs = {"step_loss": train_loss / (step // args.gradient_accumulation_steps + 1), "epoch": epoch + 1}
                        progress_bar.set_postfix(logs)
                        accelerator.log(logs, step=global_step)

                    # --- Save Checkpoint ---
                    if global_step % args.save_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"seq_model_step_{global_step}.pth")
                            accelerator.save(accelerator.unwrap_model(model).state_dict(), save_path)
                            accelerator.print(f"Saved checkpoint to {save_path}")

                # Check for max steps
                if args.max_train_steps is not None and global_step >= args.max_train_steps: break
        # End epoch loop
        if args.max_train_steps is not None and global_step >= args.max_train_steps: break

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = train_loss / num_update_steps_per_epoch
        accelerator.print(f"Epoch {epoch+1}/{args.num_epochs} finished in {epoch_time:.2f}s. Avg Loss: {avg_epoch_loss:.4f}")
        accelerator.log({"epoch_loss": avg_epoch_loss}, step=global_step)

        if accelerator.is_main_process:
            epoch_save_path = os.path.join(args.output_dir, f"seq_model_epoch_{epoch+1}.pth")
            accelerator.save(accelerator.unwrap_model(model).state_dict(), epoch_save_path)
            accelerator.print(f"Saved epoch checkpoint to {epoch_save_path}")

    accelerator.print("Training finished.")
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "seq_model_final.pth")
        accelerator.save(accelerator.unwrap_model(model).state_dict(), final_save_path)
        accelerator.print(f"Saved final model to {final_save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Noise Sequence Transformer Model")

    # Dataset Args
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the generated NPD sequence dataset directory.")
    parser.add_argument("--max_seq_len", type=int, default=11, help="Maximum sequence length to load (Source + N steps). Should match model's max_len + 1.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")

    # Model Args
    parser.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base model ID for text encoding.")
    parser.add_argument("--npnet_model_id", type=str, default="SDXL", choices=["SDXL", "DiT"], help="Specify base model type for text dim hint.") # Simplified choices
    parser.add_argument("--text_embed_dim", type=int, default=1024, help="Dimension of text embeddings used by Transformer.") # Made explicit arg
    parser.add_argument("--noise_resolution", type=int, default=128, help="Spatial resolution of noise tensors.")
    parser.add_argument("--noise_channels", type=int, default=4, help="Channels in noise tensors.")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for noise embedding.")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer internal dimension.")
    parser.add_argument("--nhead", type=int, default=8, help="Transformer attention heads.")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="Transformer text encoder layers.")
    parser.add_argument("--num_decoder_layers", type=int, default=3, help="Transformer noise decoder layers.")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="Transformer feedforward dimension.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout rate.")
    parser.add_argument("--predict_variance", action="store_true", help="Train model to predict variance (uses NLL loss).")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training.")

    # Training Args
    parser.add_argument("--output_dir", type=str, default="seq_model_training_output", help="Directory for checkpoints and logs.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max training steps (overrides epochs).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision.")
    parser.add_argument("--output_size", type=int, default=1024, help="Reference output size (used for SDXL time_ids calculation).")


    # Logging & Saving Args
    parser.add_argument("--log_steps", type=int, default=50, help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps.")

    args = parser.parse_args()

    if accelerator.is_main_process: os.makedirs(args.output_dir, exist_ok=True)
    main(args)
