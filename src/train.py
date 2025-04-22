# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import os
import time
from accelerate import Accelerator, DistributedDataParallelKwargs

# Import model and dataset loaders
try:
    from model import NPNet # Import the updated NPNet
    from dataset import get_dataloader
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure model and dataset modules are in the correct path.")
    exit(1)

# Base pipeline import
from diffusers import StableDiffusionXLPipeline, HunyuanDiTPipeline # Add others if needed

def main(args):
    # --- Initialize Accelerator ---
    # Handle potential find_unused_parameters issue with DDP when some params aren't used (e.g., conditional components)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard", # Example: Use tensorboard
        project_dir=os.path.join(args.output_dir, "logs"), # Specify log directory
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device
    # Initialize logging as early as possible
    if accelerator.is_main_process:
            accelerator.init_trackers("npnet_training") # Project name for TensorBoard/WandB

    accelerator.print(f"Using device: {device}")
    accelerator.print(f"Mixed precision: {args.mixed_precision}")
    accelerator.print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")


    # --- Load Base Model Pipeline (for prompt encoding only) ---
    accelerator.print(f"Loading base model pipeline ({args.npnet_model_id}) for text encoder...")
    # Use lower precision for base model on non-main processes to save memory? (Careful with sync)
    base_dtype = torch.float16 if args.mixed_precision != 'no' else torch.float32
    try:
        if args.npnet_model_id == 'SDXL':
            base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.base_model_id, torch_dtype=base_dtype,
                variant="fp16", use_safetensors=True, low_cpu_mem_usage=True
            )
            # Extract necessary components
            text_encoder = base_pipeline.text_encoder
            text_encoder_2 = base_pipeline.text_encoder_2
            encode_prompt = base_pipeline.encode_prompt
            get_add_time_ids = getattr(base_pipeline, "_get_add_time_ids", None)
            del base_pipeline.vae, base_pipeline.unet # Free memory
        elif args.npnet_model_id == 'DiT':
                base_pipeline = HunyuanDiTPipeline.from_pretrained(
                    args.base_model_id, torch_dtype=base_dtype, low_cpu_mem_usage=True
                )
                text_encoder = getattr(base_pipeline, "text_encoder", None)
                encode_prompt = base_pipeline.encode_prompt
                get_add_time_ids = None
                text_encoder_2 = None
                del base_pipeline.transformer, base_pipeline.vae # Free memory (adjust names for DiT)
        else:
            raise NotImplementedError(f"Base pipeline loading not implemented for {args.npnet_model_id}")

        # Move encoders to device and freeze
        if text_encoder:
                text_encoder.to(device).requires_grad_(False).eval()
        if text_encoder_2:
                text_encoder_2.to(device).requires_grad_(False).eval()

    except Exception as e:
            accelerator.print(f"Error loading base pipeline: {e}")
            exit(1)
    accelerator.print("Base model text encoder(s) loaded.")
    torch.cuda.empty_cache() # Clear cache after loading


    # --- Initialize NPNet Model ---
    accelerator.print("Initializing NPNet model...")
    model = NPNet(
        model_id=args.npnet_model_id,
        device=device, # NPNet will be moved by Accelerator later
        resolution=args.npnet_resolution,
        svd_enable_drop=args.svd_dropout,
        nt_enable_adapter=args.nt_adapter,
        nt_enable_finetune=args.nt_finetune, # Add finetune flag
        nt_enable_dropout=args.nt_dropout,   # Add dropout flag
        enable_cross_attention=args.cross_attention # Add cross-attention flag
    )

    # Load pretrained weights if specified (before prepare)
    if args.resume_from:
        accelerator.print(f"Resuming training from checkpoint: {args.resume_from}")
        # Load weights on CPU first to avoid OOM, then let prepare handle device movement
        try:
                model.load_state_dict(torch.load(args.resume_from, map_location="cpu"))
                accelerator.print("Resumed weights loaded successfully.")
        except Exception as e:
                accelerator.print(f"Error loading resume checkpoint: {e}. Starting from scratch.")


    # --- Prepare Dataset and DataLoader ---
    accelerator.print("Loading dataset...")
    metadata_path = os.path.join(args.dataset_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
            accelerator.print(f"Error: metadata.csv not found in {args.dataset_dir}")
            exit(1)
    train_dataloader = get_dataloader(
        metadata_file=metadata_path,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    if train_dataloader is None:
        accelerator.print("Failed to create DataLoader. Exiting.")
        exit(1)

    # --- Optimizer and Loss ---
    accelerator.print("Setting up optimizer and loss...")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    criterion = nn.MSELoss()

    # --- Prepare with Accelerator ---
    # Order matters: model, optimizer, dataloader
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # --- Learning Rate Scheduler (Optional) ---
    # Example: Linear warmup and decay
    # num_training_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps,
    #     num_training_steps=num_training_steps
    # )
    # lr_scheduler = accelerator.prepare(lr_scheduler) # Prepare scheduler too

    # --- Training Loop ---
    accelerator.print("Starting training...")
    # Adjust total steps calculation for gradient accumulation
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_steps = num_update_steps_per_epoch * args.num_epochs
    accelerator.print(f"Total optimization steps: {total_steps}")

    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, desc="Training Steps")
    global_step = 0

    # Resume progress bar if resuming
    # TODO: Implement state loading for global_step and progress_bar if resuming

    for epoch in range(args.num_epochs):
        model.train() # Set model to training mode
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
                if batch is None:
                    accelerator.print(f"Warning: Skipping empty batch at step {step}.")
                    continue

                # Accumulate gradients
                with accelerator.accumulate(model):
                # --- Get Data ---
                prompts = batch['prompt']
                source_noise = batch['source_noise']
                target_noise = batch['target_noise']

                # --- Encode Prompts ---
                try:
                    with torch.no_grad():
                        if args.npnet_model_id == 'SDXL':
                            prompt_embeds, _, pooled_prompt_embeds, _ = encode_prompt(
                                prompt=prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
                            )
                            # Note: NPNet currently only uses prompt_embeds
                            npnet_prompt_input = prompt_embeds
                        elif args.npnet_model_id == 'DiT':
                            prompt_embeds = encode_prompt(prompt=prompts, device=device)
                            npnet_prompt_input = prompt_embeds
                        else: # DreamShaper often SDXL based
                            prompt_embeds, _, pooled_prompt_embeds, _ = encode_prompt(
                                prompt=prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
                            )
                            npnet_prompt_input = prompt_embeds

                        # Ensure batch size matches if needed (should be handled by dataloader now)
                        if npnet_prompt_input.shape[0] != source_noise.shape[0]:
                                accelerator.print(f"Warning: Batch size mismatch between noise ({source_noise.shape[0]}) and embeds ({npnet_prompt_input.shape[0]})")
                                # Attempt to fix or skip
                                continue


                except Exception as e:
                    accelerator.print(f"Error encoding prompts at step {step}: {e}. Skipping batch.")
                    # Log error details if possible
                    accelerator.log({"prompt_encoding_error": str(e)}, step=global_step)
                    continue

                # --- Forward Pass ---
                # NPNet expects float32 internally
                predicted_noise = model(source_noise.to(torch.float32), npnet_prompt_input.to(torch.float32))

                # --- Calculate Loss ---
                loss = criterion(predicted_noise, target_noise.to(torch.float32))

                # Accumulate loss for logging (average over processes and grad steps)
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps


                # --- Backward Pass ---
                accelerator.backward(loss)

                # --- Optimizer Step ---
                if accelerator.sync_gradients:
                    # Clip gradients
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    # lr_scheduler.step() # Step scheduler if using one
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    global_step += 1

                    # --- Logging ---
                    if global_step % args.log_steps == 0:
                        logs = {
                            "step_loss": train_loss / (step + 1), # Approximate step loss
                            "lr": optimizer.param_groups[0]['lr'], # Log learning rate
                            "epoch": epoch + (step + 1) / len(train_dataloader) # Fractional epoch
                        }
                        progress_bar.set_postfix(logs)
                        accelerator.log(logs, step=global_step)
                        # Reset train_loss for next logging interval if needed, or keep accumulating for epoch loss

                    # --- Save Checkpoint ---
                    if global_step % args.save_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"npnet_step_{global_step}.pth")
                            # Save unwrapped model state dict
                            unwrapped_model = accelerator.unwrap_model(model)
                            torch.save(unwrapped_model.state_dict(), save_path)
                            accelerator.print(f"Saved checkpoint to {save_path}")

                # End of accumulation block

                # Check if max steps reached
                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    accelerator.print(f"Reached max_train_steps ({args.max_train_steps}). Stopping training.")
                    break
        # End of epoch loop
        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break # Exit outer loop too

        avg_epoch_loss = train_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch+1}/{args.num_epochs} finished. Average Loss: {avg_epoch_loss:.4f}")
        accelerator.log({"epoch_loss": avg_epoch_loss}, step=global_step) # Log epoch loss

        # Save model at the end of each epoch
        if accelerator.is_main_process:
            epoch_save_path = os.path.join(args.output_dir, f"npnet_epoch_{epoch+1}.pth")
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), epoch_save_path)
            accelerator.print(f"Saved epoch checkpoint to {epoch_save_path}")


    accelerator.print("Training finished.")
    # Save final model
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "npnet_final.pth")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), final_save_path)
        accelerator.print(f"Saved final model to {final_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NPNet Model using Accelerate")

    # Dataset Args
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the generated NPD dataset directory.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")

    # Model Args
    parser.add_argument("--npnet_model_id", type=str, default="SDXL", choices=["SDXL", "DreamShaper", "DiT"], help="Base model type NPNet is for.")
    parser.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base model ID for text encoding.")
    parser.add_argument("--npnet_resolution", type=int, default=128, help="Internal resolution for NPNet U-Nets.")
    parser.add_argument("--svd_dropout", action="store_true", help="Enable dropout/layernorm in SVDNoiseUnet.")
    parser.add_argument("--nt_adapter", action="store_true", help="Enable adapter in NoiseTransformer.")
    parser.add_argument("--nt_finetune", action="store_true", help="Enable finetuning last 2 stages in NoiseTransformer's Swin.")
    parser.add_argument("--nt_dropout", action="store_true", default=True, help="Enable dropout in NoiseTransformer (default: True).") # Default based on code
    parser.add_argument("--cross_attention", action="store_true", help="Enable cross-attention in NPNet.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training.")

    # Training Args
    parser.add_argument("--output_dir", type=str, default="npnet_training_output", help="Directory for checkpoints and logs.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps to accumulate gradients.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision.")
    # parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type.")
    # parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for LR scheduler.")

    # Logging & Saving Args
    parser.add_argument("--log_steps", type=int, default=100, help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps.")
    parser.add_argument("--output_size", type=int, default=1024, help="Reference output size for SDXL time_ids.")


    args = parser.parse_args()

    # Create output directory if it doesn't exist (only on main process)
    if Accelerator().is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)

    main(args)