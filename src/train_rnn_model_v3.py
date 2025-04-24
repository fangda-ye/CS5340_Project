# src/train_rnn_model_v3.py
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
import pandas as pd
import shutil # For removing old checkpoints

# Import IMPROVED RNN model (v3) and sequence dataset loader
try:
    from model import NoiseSequenceRNN_v3 # Import the V3 RNN model
    from dataset import get_sequence_dataloader
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure model (rnn_seq_model_v3.py) and dataset modules are in the correct path.")
    exit(1)

# Base pipeline import
from diffusers import StableDiffusionXLPipeline, HunyuanDiTPipeline

# --- Loss Functions ---
def gaussian_nll_loss(mean_pred, log_var_pred, target, reduction='mean'):
    log_var_pred = torch.clamp(log_var_pred, min=-10, max=10); inv_var = torch.exp(-log_var_pred)
    mse_term = (target - mean_pred)**2 * inv_var; loss = 0.5 * (mse_term + log_var_pred)
    if reduction == 'mean': return loss.mean()
    elif reduction == 'sum': return loss.sum()
    else: return loss

def kl_divergence_normal_standard_normal(mean, log_var, reduction='mean'):
    variance = torch.exp(log_var); kl_div = 0.5 * (variance + mean.pow(2) - 1.0 - log_var)
    if reduction == 'mean': return kl_div.mean()
    elif reduction == 'sum': return kl_div.sum()
    else: return kl_div

# --- Main Training Function ---
def main(args):
    # --- Initialize Accelerator ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard", project_dir=os.path.join(args.output_dir, "logs"),
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device
    if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            accelerator.init_trackers("rnn_v3_seq_model_training")
    accelerator.print(f"Using device: {device}, Mixed precision: {args.mixed_precision}")
    accelerator.print(f"Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # --- Load Base Model Pipeline (for text encoding) ---
    # (Same logic as before - ensure encode_text returns pooled embedding [B, D_txt])
    accelerator.print(f"Loading base model pipeline ({args.base_model_id}) for text encoder...")
    base_dtype = torch.float16 if args.mixed_precision != 'no' else torch.float32
    text_encoder, text_encoder_2 = None, None
    try:
        if "xl" in args.base_model_id.lower() and "hunyuan" not in args.base_model_id.lower():
            pipe_class = StableDiffusionXLPipeline
            base_pipeline = pipe_class.from_pretrained(args.base_model_id, torch_dtype=base_dtype, variant="fp16", use_safetensors=True, low_cpu_mem_usage=True)
            text_encoder = base_pipeline.text_encoder; text_encoder_2 = base_pipeline.text_encoder_2
            def encode_text(prompt_list):
                    with torch.no_grad(): _, _, pooled_prompt_embeds, _ = base_pipeline.encode_prompt(prompt=prompt_list, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False); return pooled_prompt_embeds
            del base_pipeline.vae, base_pipeline.unet
        elif "hunyuan" in args.base_model_id.lower():
                pipe_class = HunyuanDiTPipeline
                base_pipeline = pipe_class.from_pretrained(args.base_model_id, torch_dtype=base_dtype, low_cpu_mem_usage=True)
                text_encoder = getattr(base_pipeline, "text_encoder", None)
                def encode_text(prompt_list):
                    with torch.no_grad(): embed_output = base_pipeline.encode_prompt(prompt=prompt_list, device=device); return embed_output # Adjust
                del base_pipeline.transformer, base_pipeline.vae; text_encoder_2 = None
        else: raise NotImplementedError(f"Base pipeline loading not implemented for {args.base_model_id}")
        if text_encoder: text_encoder.to(device).requires_grad_(False).eval()
        if text_encoder_2: text_encoder_2.to(device).requires_grad_(False).eval()
    except Exception as e: accelerator.print(f"Error loading base pipeline for text encoding: {e}"); exit(1)
    accelerator.print("Base model text encoder(s) loaded.")
    torch.cuda.empty_cache()

    # --- Initialize Noise Sequence RNN Model (V3) ---
    accelerator.print("Initializing NoiseSequenceRNN_v3 model...")
    if args.npnet_model_id == 'DiT': text_embed_dim = 1024
    else: text_embed_dim = 1280 # SDXL pooled
    if args.text_embed_dim > 0: text_embed_dim = args.text_embed_dim

    model = NoiseSequenceRNN_v3(
        text_embed_dim=text_embed_dim, noise_img_size=args.noise_resolution,
        noise_in_chans=args.noise_channels, cnn_base_filters=args.cnn_base_filters,
        cnn_num_blocks_per_stage=args.cnn_num_blocks, cnn_feat_dim=args.cnn_feat_dim,
        cnn_groups=args.cnn_groups, gru_hidden_size=args.gru_hidden_size,
        gru_num_layers=args.gru_num_layers, gru_dropout=args.gru_dropout,
        predict_variance=args.predict_variance
        # predict_residual is removed as we predict full state now
    )

    if args.resume_from:
        accelerator.print(f"Resuming training from checkpoint: {args.resume_from}")
        try: model.load_state_dict(torch.load(args.resume_from, map_location="cpu"))
        except Exception as e: accelerator.print(f"Error loading resume checkpoint: {e}.")

    # --- Prepare Dataset and DataLoader ---
    accelerator.print("Loading sequence dataset...")
    metadata_path = os.path.join(args.dataset_dir, "metadata.csv")
    if not os.path.exists(metadata_path): accelerator.print(f"Error: metadata.csv not found in {args.dataset_dir}"); exit(1)
    try:
            meta_df_peek = pd.read_csv(metadata_path, nrows=1); dataset_num_steps = meta_df_peek['num_steps'].iloc[0]
            dataloader_max_seq = dataset_num_steps; print(f"Detected sequence length from metadata: {dataset_num_steps}")
    except Exception as e: print(f"Warning: Could not read num_steps from metadata ({e}). Using default max_seq_len=10."); dataloader_max_seq = 10
    train_dataloader = get_sequence_dataloader(
        metadata_file=metadata_path, dataset_dir=args.dataset_dir, batch_size=args.batch_size,
        max_seq_len=dataloader_max_seq, shuffle=True, num_workers=args.num_workers
    )
    if train_dataloader is None: accelerator.print("Failed to create DataLoader."); exit(1)

    # --- Optimizer and Loss ---
    accelerator.print("Setting up optimizer and loss...")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    criterion = gaussian_nll_loss if args.predict_variance else nn.MSELoss()
    loss_type = "NLL" if args.predict_variance else "MSE"
    print(f"Using {loss_type} Loss.")
    print(f"Model is predicting full state.")
    if args.kl_weight > 0: print(f"Using KL divergence regularization with weight {args.kl_weight}")

    # --- Prepare with Accelerator ---
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # --- Training Loop ---
    accelerator.print("Starting training...")
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * args.num_epochs
    if args.max_train_steps: total_steps = min(args.max_train_steps, total_steps)
    accelerator.print(f"Total optimization steps: {total_steps}")
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, desc="Training Steps")
    global_step = 0
    # Keep track of saved checkpoints to manage disk space
    saved_checkpoints = []

    for epoch in range(args.num_epochs):
        model.train(); train_loss = 0.0; train_nll_loss = 0.0; train_kl_loss = 0.0
        epoch_start_time = time.time()

        for step, batch in enumerate(train_dataloader):
                if batch is None: accelerator.print(f"Warning: Skipping empty batch at step {step}."); continue

                with accelerator.accumulate(model):
                    prompts = batch['prompt']; source_noise = batch['source_noise']; golden_sequence = batch['golden_sequence']
                    B, T_target, C, H, W = golden_sequence.shape
                    input_noise_seq = torch.cat([source_noise.unsqueeze(1), golden_sequence[:, :-1]], dim=1)
                    target_noise_seq = golden_sequence # Target is the full sequence x'_1 to x'_n

                try:
                    with torch.no_grad(): text_embed = encode_text(prompts).to(device)
                except Exception as e: accelerator.print(f"Error encoding prompts: {e}"); continue

                input_noise_seq_f32 = input_noise_seq.to(torch.float32)
                text_embed_f32 = text_embed.to(torch.float32)
                target_noise_seq_f32 = target_noise_seq.to(torch.float32)

                # --- Calculate Primary Loss (NLL or MSE) ---
                loss = 0.0; nll_loss_val = 0.0; kl_loss_val = 0.0

                if args.predict_variance:
                    mean_pred_seq, log_var_pred_seq, _ = model(input_noise_seq_f32, text_embed_f32)
                    nll_loss = criterion(mean_pred_seq, log_var_pred_seq, target_noise_seq_f32)
                    loss += nll_loss
                    nll_loss_val = nll_loss.item()

                    # --- Calculate KL Divergence Regularization (Optional) ---
                    if args.kl_weight > 0:
                        # KL divergence between predicted N(mu_k, sigma_k^2) and N(0, I)
                        kl_loss = kl_divergence_normal_standard_normal(mean_pred_seq, log_var_pred_seq)
                        loss += args.kl_weight * kl_loss
                        kl_loss_val = kl_loss.item()
                else:
                    mean_pred_seq, _ = model(input_noise_seq_f32, text_embed_f32)
                    loss = criterion(mean_pred_seq, target_noise_seq_f32)
                    nll_loss_val = loss.item()

                # Accumulate loss values
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                avg_nll_loss = accelerator.gather(torch.tensor(nll_loss_val, device=device).repeat(args.batch_size)).mean()
                train_nll_loss += avg_nll_loss.item() / args.gradient_accumulation_steps
                if args.kl_weight > 0:
                        avg_kl_loss = accelerator.gather(torch.tensor(kl_loss_val, device=device).repeat(args.batch_size)).mean()
                        train_kl_loss += avg_kl_loss.item() / args.gradient_accumulation_steps

                # Backward Pass & Optimize
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step(); optimizer.zero_grad()
                    progress_bar.update(1); global_step += 1

                    # Logging
                    if global_step % args.log_steps == 0:
                        num_opt_steps_so_far = step // args.gradient_accumulation_steps + 1
                        logs = {"step_loss": train_loss / num_opt_steps_so_far, "nll_loss": train_nll_loss / num_opt_steps_so_far, "epoch": epoch + 1}
                        if args.kl_weight > 0: logs["kl_loss"] = train_kl_loss / num_opt_steps_so_far
                        progress_bar.set_postfix(logs); accelerator.log(logs, step=global_step)

                    # --- Save Checkpoint (Less Frequently) ---
                    if global_step % args.save_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"rnn_v3_model_step_{global_step}.pth")
                            accelerator.save(accelerator.unwrap_model(model).state_dict(), save_path)
                            accelerator.print(f"Saved checkpoint to {save_path}")
                            # --- Manage Checkpoints ---
                            saved_checkpoints.append(save_path)
                            if args.max_checkpoints > 0 and len(saved_checkpoints) > args.max_checkpoints:
                                    oldest_ckpt = saved_checkpoints.pop(0)
                                    try:
                                        os.remove(oldest_ckpt)
                                        accelerator.print(f"Removed oldest checkpoint: {oldest_ckpt}")
                                    except OSError as e:
                                        accelerator.print(f"Error removing oldest checkpoint {oldest_ckpt}: {e}")

                if args.max_train_steps is not None and global_step >= args.max_train_steps: break
        if args.max_train_steps is not None and global_step >= args.max_train_steps: break

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = train_loss / num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0.0
        avg_epoch_nll = train_nll_loss / num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0.0
        avg_epoch_kl = train_kl_loss / num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0.0
        accelerator.print(f"Epoch {epoch+1}/{args.num_epochs} finished in {epoch_time:.2f}s. Avg Loss: {avg_epoch_loss:.4f} (NLL: {avg_epoch_nll:.4f}, KL: {avg_epoch_kl:.4f})")
        log_epoch = {"epoch_loss": avg_epoch_loss, "epoch_nll": avg_epoch_nll}
        if args.kl_weight > 0: log_epoch["epoch_kl"] = avg_epoch_kl
        accelerator.log(log_epoch, step=global_step)

        # --- Save End-of-Epoch Checkpoint (Optional, less frequent than step saves) ---
        # Consider removing epoch saves if save_steps is frequent enough to save space
        # if accelerator.is_main_process:
        #     epoch_save_path = os.path.join(args.output_dir, f"rnn_v3_model_epoch_{epoch+1}.pth")
        #     accelerator.save(accelerator.unwrap_model(model).state_dict(), epoch_save_path)
        #     accelerator.print(f"Saved epoch checkpoint to {epoch_save_path}")


    accelerator.print("Training finished.")
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "rnn_v3_model_final.pth")
        accelerator.save(accelerator.unwrap_model(model).state_dict(), final_save_path)
        accelerator.print(f"Saved final model to {final_save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Improved Noise Sequence RNN Model (v3)")
    # Dataset Args
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to NPD sequence dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    # Model Args
    parser.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base model ID for text encoding.")
    parser.add_argument("--npnet_model_id", type=str, default="SDXL", choices=["SDXL", "DiT"], help="Base model type for text dim hint.")
    parser.add_argument("--text_embed_dim", type=int, default=0, help="Dimension of text embeddings (0 to infer).")
    parser.add_argument("--noise_resolution", type=int, default=128, help="Spatial resolution of noise.")
    parser.add_argument("--noise_channels", type=int, default=4, help="Channels in noise.")
    parser.add_argument("--cnn_base_filters", type=int, default=64, help="Base filters for ResNet CNN.")
    parser.add_argument("--cnn_num_blocks", type=int, nargs='+', default=[2, 2, 2, 2], help="List: Num ResBlocks per stage.")
    parser.add_argument("--cnn_feat_dim", type=int, default=512, help="Feature dim from CNN encoder.")
    parser.add_argument("--cnn_groups", type=int, default=8, help="Num groups for GroupNorm.")
    parser.add_argument("--gru_hidden_size", type=int, default=1024, help="GRU hidden size.")
    parser.add_argument("--gru_num_layers", type=int, default=2, help="Number of GRU layers.")
    parser.add_argument("--gru_dropout", type=float, default=0.1, help="GRU dropout.")
    parser.add_argument("--predict_variance", action="store_true", help="Train model to predict variance (uses NLL loss).")
    # Removed predict_residual flag as V3 predicts full state
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint.")
    # Training Args
    parser.add_argument("--output_dir", type=str, default="rnn_v3_seq_model_output", help="Output directory.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max training steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU.") # Reduced default
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation.") # Increased default
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision.")
    parser.add_argument("--kl_weight", type=float, default=0.01, help="Weight for KL divergence loss (0 to disable).")
    parser.add_argument("--output_size", type=int, default=1024, help="Reference output size for SDXL time_ids.")
    # Logging & Saving Args
    parser.add_argument("--log_steps", type=int, default=50, help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps.")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="Maximum number of checkpoints to keep (0 for all).")


    args = parser.parse_args()
    # Convert cnn_num_blocks from string list if needed
    if isinstance(args.cnn_num_blocks, list) and len(args.cnn_num_blocks) > 0 and isinstance(args.cnn_num_blocks[0], str):
            try: args.cnn_num_blocks = [int(b) for b in args.cnn_num_blocks]
            except ValueError: print("Error: --cnn_num_blocks must be integers."); exit(1)

    main(args)
