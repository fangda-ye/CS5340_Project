# model/seq_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# --- Helper Modules ---

class PositionalEncoding(nn.Module):
    """ Standard sinusoidal positional encoding """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # Ensure correct slicing for odd/even dimensions
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            # Handle odd d_model dimension if necessary, though typically even
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:d_model//2]
        else:
                pe[:, 0, 1::2] = torch.cos(position * div_term)

        pe = pe.squeeze(1).unsqueeze(0) # [1, max_len, d_model] for batch_first=True
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding up to the sequence length of x
        if x.size(1) > self.pe.size(1):
                raise ValueError(f"Sequence length {x.size(1)} exceeds max_len {self.pe.size(1)} in PositionalEncoding")
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    """ Embed patches of input noise tensor """
    def __init__(self, img_size=128, patch_size=16, in_chans=4, embed_dim=768):
        super().__init__()
        # Handle potential non-square images if needed, assume square for now
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        # Convolutional layer to create patches and project to embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
                # Option 1: Raise error
                # raise ValueError(f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
                # Option 2: Resize input (might affect results)
                print(f"Warning: Resizing PatchEmbedding input from {H}x{W} to {self.img_size[0]}x{self.img_size[1]}")
                x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)


        # Project and flatten: [B, C, H, W] -> [B, E, H/P, W/P] -> [B, E, N] -> [B, N, E]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x # Output shape: [Batch, NumPatches, EmbedDim]

# --- Main Sequence Transformer Model ---

class NoiseSequenceTransformer(nn.Module):
    """
    Transformer model to predict a sequence of evolving noise tensors autoregressively,
    conditioned on a text prompt embedding.
    Uses a standard Transformer Encoder-Decoder architecture.
    """
    def __init__(self,
                    # Text Encoder Config
                    text_embed_dim: int,
                    transformer_d_model: int,
                    transformer_nhead: int,
                    transformer_num_encoder_layers: int,
                    transformer_num_decoder_layers: int,
                    transformer_dim_feedforward: int,
                    transformer_dropout: float = 0.1,
                    # Noise Input/Output Config
                    noise_img_size: int = 128,
                    noise_patch_size: int = 16,
                    noise_in_chans: int = 4,
                    # Output Config
                    predict_variance: bool = True,
                    max_seq_len: int = 15 # Max steps in noise sequence + 1 (for initial noise)
                ):
        super().__init__()

        self.transformer_d_model = transformer_d_model
        self.predict_variance = predict_variance
        self.noise_img_size = noise_img_size
        self.noise_patch_size = noise_patch_size
        self.noise_in_chans = noise_in_chans

        # --- Text Embedding Projection ---
        self.text_projection = nn.Linear(text_embed_dim, transformer_d_model) if text_embed_dim != transformer_d_model else nn.Identity()

        # --- Noise Patch Embedding ---
        self.patch_embed = PatchEmbedding(
            img_size=noise_img_size, patch_size=noise_patch_size,
            in_chans=noise_in_chans, embed_dim=transformer_d_model
        )
        num_patches = self.patch_embed.num_patches

        # --- Positional Encodings ---
        self.pos_encoder_noise = PositionalEncoding(transformer_d_model, transformer_dropout, max_len=num_patches)
        # Assuming text embedding might be sequential, add PE for it too
        # Max text sequence length needed if text_embed is sequential [B, SeqTxt, Dim]
        # If text_embed is pooled [B, Dim], PE is not applied in the standard way here.
        # Let's assume text PE is handled externally or text is pooled.
        # self.pos_encoder_text = PositionalEncoding(transformer_d_model, transformer_dropout, max_len=max_text_len)

        # --- Transformer ---
        # Using nn.Transformer which contains Encoder and Decoder
        self.transformer = nn.Transformer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True # Crucial: Input shapes are [Batch, SeqLen, Dim]
        )

        # --- Output Head ---
        # Project decoder output features back to patch representation
        # Output needs to represent the parameters (mean or mean+logvar) for each patch
        output_patch_channels = noise_in_chans * 2 if predict_variance else noise_in_chans
        # The projection needs to map d_model back to the flattened patch dimension C*P*P
        # This seems overly complex. Let's rethink the output.

        # --- Alternative Output Head: Project features then reconstruct ---
        # Project each output patch feature vector [B, NumPatches, D_model] -> [B, NumPatches, OutputPatchDim]
        output_patch_dim_flat = noise_in_chans * noise_patch_size * noise_patch_size
        output_head_dim_flat = output_patch_dim_flat * 2 if predict_variance else output_patch_dim_flat
        self.output_projection = nn.Linear(transformer_d_model, output_head_dim_flat)

        print(f"Initialized NoiseSequenceTransformer:")
        print(f"  - Transformer d_model: {transformer_d_model}, nhead: {transformer_nhead}")
        print(f"  - Encoder Layers: {transformer_num_encoder_layers}, Decoder Layers: {transformer_num_decoder_layers}")
        print(f"  - Noise Patches: {num_patches} ({self.patch_embed.grid_size}), Patch Size: {noise_patch_size}")
        print(f"  - Predict Variance: {predict_variance}")


    def _generate_square_subsequent_mask(self, sz, device):
        """Generates a square causal mask for the sequence. sz: sequence length."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1) # Upper triangle, diagonal is 0
        mask = mask.masked_fill(mask == 1, float('-inf')) # Fill upper triangle with -inf
        # mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float('-inf'))
        return mask # Shape [sz, sz]

    def forward(self, src_noise_sequence, text_embed):
        """
        Forward pass for training (using teacher forcing).

        Args:
            src_noise_sequence (torch.Tensor): Sequence of input noise tensors
                                                [B, SeqLen_in, C, H, W].
                                                Example: [x_T, x'_1, ..., x'_{n-1}]
            text_embed (torch.Tensor): Text embedding [B, SeqLen_txt, D_txt] or [B, D_txt].

        Returns:
            If predict_variance:
                tuple(torch.Tensor, torch.Tensor): Predicted mean and log_variance sequences
                                                    [B, SeqLen_out, C, H, W]. SeqLen_out = SeqLen_in.
            Else:
                torch.Tensor: Predicted mean sequence [B, SeqLen_out, C, H, W].
        """
        B, T_in, C, H, W = src_noise_sequence.shape
        device = src_noise_sequence.device

        # --- 1. Encode Text ---
        projected_text_embed = self.text_projection(text_embed) # [B, SeqLen_txt, D_model] or [B, D_model]
        if projected_text_embed.dim() == 2:
                projected_text_embed = projected_text_embed.unsqueeze(1) # Assume [B, 1, D_model] if pooled
        # TODO: Add text positional encoding if needed
        # memory = self.transformer_encoder(projected_text_embed) # Use internal encoder
        # Using full nn.Transformer, memory is calculated from text_embed as source
        memory = self.transformer.encoder(projected_text_embed) # [B, SeqLen_txt, D_model]


        # --- 2. Prepare Decoder Input Noise Sequence ---
        # Embed all input noises in the sequence
        noise_flat_batch = src_noise_sequence.view(B * T_in, C, H, W)
        noise_patches_embedded = self.patch_embed(noise_flat_batch) # [B * T_in, NumPatches, D_model]
        noise_patches_embedded = self.pos_encoder_noise(noise_patches_embedded) # Apply PE to patches
        # Reshape for decoder input: [B, T_in * NumPatches, D_model]? No, decoder expects target sequence.
        # Decoder target sequence should be [B, SeqLen_tgt, D_model]
        # Here, SeqLen_tgt corresponds to the sequence of noise patches we want to predict.
        # Let's process the sequence step-by-step conceptually, but implement vectorized if possible.

        # --- Vectorized Decoder Input Preparation ---
        # Reshape embedded patches: [B, T_in, NumPatches, D_model]
        decoder_input_embedded = noise_patches_embedded.view(B, T_in, self.patch_embed.num_patches, self.transformer_d_model)
        # Flatten patches into sequence dim: [B, T_in * NumPatches, D_model]
        # This treats the whole sequence of patches as the target sequence.
        num_patches = self.patch_embed.num_patches
        decoder_input_flat = decoder_input_embedded.view(B, T_in * num_patches, self.transformer_d_model)

        # --- 3. Create Masks ---
        # Decoder target mask (causal mask) for the flattened sequence of patches
        tgt_len = T_in * num_patches
        decoder_tgt_mask = self._generate_square_subsequent_mask(tgt_len, device) # [TgtLen, TgtLen]

        # --- 4. Pass through Transformer Decoder ---
        # Input `tgt` to decoder should be the sequence we want predictions for, shifted or masked.
        # Standard transformer usage: tgt is the target sequence (e.g., ground truth shifted right).
        # Here, decoder_input_flat represents the input sequence [xT_p1..N, x1_p1..N, ..., xN-1_p1..N]
        # The decoder should predict the *next* patch in the sequence based on previous patches and memory.
        decoder_output = self.transformer.decoder(
            tgt=decoder_input_flat, # Input sequence [B, TgtLen, D_model]
            memory=memory,          # Text context [B, SrcLen, D_model]
            tgt_mask=decoder_tgt_mask # Causal mask [TgtLen, TgtLen]
            # memory_mask=None      # Optional mask for memory
        ) # Output shape: [B, TgtLen, D_model]

        # --- 5. Project to Output Noise Parameters ---
        # Project features of each patch to the flattened patch parameters (mean or mean+logvar)
        projected_output_flat = self.output_projection(decoder_output) # [B, TgtLen, OutputPatchDimFlat]

        # --- 6. Reshape and Reconstruct Noise Sequence ---
        # Reshape back to sequence of patches: [B, T_in, NumPatches, OutputPatchDimFlat]
        output_patch_dim_flat = self.noise_in_chans * self.noise_patch_size * self.noise_patch_size
        output_channels_per_patch = output_patch_dim_flat * 2 if self.predict_variance else output_patch_dim_flat
        projected_output_seq = projected_output_flat.view(B, T_in, num_patches, output_channels_per_patch)

        # Reconstruct each noise tensor in the sequence
        # This requires un-patching. We projected to flattened patches, need to reshape and 'unfold'.
        # This is the inverse of PatchEmbedding.proj. A ConvTranspose2d is simpler.

        # --- Alternative Reconstruction using ConvTranspose ---
        # Reshape decoder output: [B, TgtLen, D_model] -> [B*T_in, NumPatches, D_model]
        decoder_output_reshaped = decoder_output.view(B * T_in, num_patches, self.transformer_d_model)
        # Reshape for ConvTranspose: [B*T_in, D_model, H_grid, W_grid]
        H_grid = W_grid = self.patch_embed.grid_size[0]
        decoder_output_spatial = decoder_output_reshaped.transpose(1, 2).view(B * T_in, self.transformer_d_model, H_grid, W_grid)

        # Reconstruct noise parameters using ConvTranspose
        # Input: [B*T_in, D_model, H_grid, W_grid]
        # Output: [B*T_in, final_out_channels, H, W]
        final_out_channels = self.noise_in_chans * 2 if self.predict_variance else self.noise_in_chans
        # Re-initialize reconstruction layer here based on final_out_channels
        self.noise_reconstruction = nn.ConvTranspose2d(
            self.transformer_d_model, final_out_channels,
            kernel_size=self.noise_patch_size, stride=self.noise_patch_size
        ).to(device) # Ensure it's on the right device

        reconstructed_params_flat = self.noise_reconstruction(decoder_output_spatial)

        # Reshape back to sequence: [B, T_in, final_out_channels, H, W]
        output_sequence = reconstructed_params_flat.view(B, T_in, final_out_channels, H, W)

        # --- 7. Separate Mean and Log Variance ---
        if self.predict_variance:
            # Split along channel dimension (dim=2)
            mean_seq, log_var_seq = torch.chunk(output_sequence, 2, dim=2)
            log_var_seq = torch.clamp(log_var_seq, min=-10, max=10) # Clamp log variance
            return mean_seq, log_var_seq
        else:
            return output_sequence # Only mean predicted

