# model/rnn_seq_model_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# --- FiLM Layer ---
class FiLMLayer(nn.Module):
    """Applies Feature-wise Linear Modulation"""
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        # gamma and beta are predicted from text_embed
        # Shape needs to match x channels: [B, C, 1, 1] for spatial modulation
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # Add spatial dims: [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)   # Add spatial dims: [B, C, 1, 1]
        return gamma * x + beta

# --- ResNet Style Block ---
class ResBlock(nn.Module):
    """ Simple Residual Block with GroupNorm and SiLU activation """
    def __init__(self, in_channels, out_channels, groups=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels)
        self.act1 = nn.SiLU() # Swish activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Shortcut connection if channels change
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual

# --- Enhanced CNN Encoder ---
class ResNetCNNEncoder(nn.Module):
    """ ResNet style CNN to encode noise [B, C, H, W] -> [B, D_feat] """
    def __init__(self, in_chans=4, base_filters=32, num_blocks_per_stage=[2, 2, 2], feat_dim=512, groups=8):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_chans, base_filters, kernel_size=3, padding=1)
        current_chans = base_filters
        self.stages = nn.ModuleList()

        for i, num_blocks in enumerate(num_blocks_per_stage):
            stage_layers = []
            out_chans = base_filters * (2**i)
            # Downsample at the start of each stage (except first)
            if i > 0:
                stage_layers.append(nn.Conv2d(current_chans, out_chans, kernel_size=3, stride=2, padding=1))
            else: # First stage might just change channels without downsampling
                    stage_layers.append(nn.Conv2d(current_chans, out_chans, kernel_size=3, stride=1, padding=1))

            current_chans = out_chans
            for _ in range(num_blocks):
                stage_layers.append(ResBlock(current_chans, current_chans, groups=groups))
            self.stages.append(nn.Sequential(*stage_layers))

        self.final_norm = nn.GroupNorm(num_groups=min(groups, current_chans), num_channels=current_chans)
        self.final_act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(current_chans, feat_dim)

    def forward(self, x):
        x = self.initial_conv(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# --- Enhanced CNN Decoder with FiLM ---
class ResNetCNNDecoderFiLM(nn.Module):
    """ ResNet style CNN Decoder with FiLM conditioning (Corrected FiLM application) """
    def __init__(self, feat_dim=512, text_embed_dim=1280, target_chans=4, target_size=128,
                    base_filters=32, num_blocks_per_stage=[2, 2, 2], groups=8):
        super().__init__()
        self.target_chans = target_chans
        self.target_size = target_size

        num_stages = len(num_blocks_per_stage)
        initial_chans = base_filters * (2**(num_stages - 1))
        initial_size = target_size // (2**(num_stages - 1))
        if initial_size < 1: initial_size = 1

        self.initial_fc = nn.Linear(feat_dim, initial_chans * initial_size * initial_size)
        self.initial_reshape_size = (initial_chans, initial_size, initial_size)

        self.res_blocks = nn.ModuleList()
        self.film_generators = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.film_layer = FiLMLayer() # Instantiate FiLM layer

        current_chans = initial_chans
        for i in range(num_stages - 1, -1, -1): # Iterate backwards
            # --- ResBlocks for this stage ---
            stage_res_blocks = []
            for _ in range(num_blocks_per_stage[i]):
                stage_res_blocks.append(ResBlock(current_chans, current_chans, groups=groups))
            self.res_blocks.append(nn.Sequential(*stage_res_blocks)) # Add ResBlocks for this stage

            # --- FiLM Generator for this stage's output ---
            # Generates params based on current_chans (output of ResBlocks)
            film_out_dim = current_chans * 2 # gamma and beta
            self.film_generators.append(nn.Linear(text_embed_dim, film_out_dim))

            # --- Upsampler (ConvTranspose) for this stage ---
            # Upsample channels for the *next* stage (i-1) or final output
            out_chans = base_filters * (2**(i-1)) if i > 0 else base_filters
            if i > 0: # Add upsampler unless it's the last stage before final conv
                    self.upsamplers.append(nn.ConvTranspose2d(current_chans, out_chans, kernel_size=4, stride=2, padding=1))
                    current_chans = out_chans # Update channels for the next stage input
            else:
                    # Keep current_chans for the final convolution
                    self.upsamplers.append(nn.Identity()) # No upsampling needed here

        # Final convolution to get target channels
        self.final_conv = nn.Conv2d(current_chans, target_chans, kernel_size=3, padding=1)


    def forward(self, x, text_embed):
        # Project and reshape feature vector from GRU hidden state
        x = self.initial_fc(x)
        x = x.view(-1, *self.initial_reshape_size) # [B, C_init, H_init, W_init]

        # Iterate through stages (ResBlocks -> FiLM -> Upsample)
        for i in range(len(self.res_blocks)):
            # Apply ResBlocks
            x = self.res_blocks[i](x)

            # Apply FiLM using corresponding generator
            film_params = self.film_generators[i](text_embed) # [B, C*2]
            gamma, beta = torch.chunk(film_params, 2, dim=-1) # [B, C], [B, C]
            x = self.film_layer(x, gamma, beta) # Modulate features

            # Apply Upsampler
            x = self.upsamplers[i](x)

        # Final convolution
        x = self.final_conv(x)

        # Ensure output size matches target (optional interpolation)
        if x.shape[-2:] != (self.target_size, self.target_size):
                x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

        return x

# # --- Enhanced CNN Decoder with FiLM ---
# class ResNetCNNDecoderFiLM(nn.Module):
#     """ ResNet style CNN Decoder with FiLM conditioning """
#     def __init__(self, feat_dim=512, text_embed_dim=1280, target_chans=4, target_size=128,
#                     base_filters=32, num_blocks_per_stage=[2, 2, 2], groups=8):
#         super().__init__()
#         self.target_chans = target_chans
#         self.target_size = target_size

#         # Calculate initial channels and spatial size for ConvTranspose
#         num_stages = len(num_blocks_per_stage)
#         initial_chans = base_filters * (2**(num_stages - 1))
#         initial_size = target_size // (2**(num_stages - 1)) # Size after last downsample in encoder
#         if initial_size < 1: initial_size = 1

#         self.initial_fc = nn.Linear(feat_dim, initial_chans * initial_size * initial_size)
#         self.initial_reshape_size = (initial_chans, initial_size, initial_size)

#         self.stages = nn.ModuleList()
#         self.film_generators = nn.ModuleList()
#         current_chans = initial_chans

#         for i in range(num_stages - 1, -1, -1): # Iterate backwards
#             stage_layers = []
#             out_chans = base_filters * (2**(i-1)) if i > 0 else base_filters # Channels before upsampling

#             # Add ResBlocks first
#             for _ in range(num_blocks_per_stage[i]):
#                 stage_layers.append(ResBlock(current_chans, current_chans, groups=groups))

#             # Upsample (except for the last stage output)
#             if i > 0:
#                     stage_layers.append(nn.ConvTranspose2d(current_chans, out_chans, kernel_size=4, stride=2, padding=1))
#             else: # Last stage before final conv
#                     out_chans = current_chans # No channel change before final conv

#             self.stages.append(nn.Sequential(*stage_layers))

#             # FiLM generator for the *output* of this stage (or input to next ResBlock stage)
#             # Modulate channels *before* upsampling or final conv
#             film_out_dim = current_chans * 2 # gamma and beta
#             self.film_generators.append(nn.Linear(text_embed_dim, film_out_dim))

#             current_chans = out_chans # Update channels for next stage

#         # Final convolution to get target channels
#         self.final_conv = nn.Conv2d(current_chans, target_chans, kernel_size=3, padding=1)
#         self.film_layer = FiLMLayer()

#     def forward(self, x, text_embed):
#         # Project and reshape feature vector from GRU hidden state
#         x = self.initial_fc(x)
#         x = x.view(-1, *self.initial_reshape_size) # [B, C_init, H_init, W_init]

#         stage_idx = 0
#         for stage in self.stages:
#             x = stage(x)
#             # Apply FiLM using corresponding generator
#             if stage_idx < len(self.film_generators):
#                     film_params = self.film_generators[stage_idx](text_embed) # [B, C*2]
#                     gamma, beta = torch.chunk(film_params, 2, dim=-1) # [B, C], [B, C]
#                     x = self.film_layer(x, gamma, beta)
#                     stage_idx += 1


#         # Final convolution
#         x = self.final_conv(x)

#         # Ensure output size matches target (optional interpolation)
#         if x.shape[-2:] != (self.target_size, self.target_size):
#                 x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

#         return x

# --- Main Improved RNN Sequence Model ---
class NoiseSequenceRNN_v2(nn.Module):
    """
    Improved RNN (GRU) based model using ResNet CNNs and FiLM conditioning.
    Optionally predicts residuals and includes KL divergence loss support.
    """
    def __init__(self,
                    # Text Embed Config
                    text_embed_dim: int,
                    # Noise I/O Config
                    noise_img_size: int = 128,
                    noise_in_chans: int = 4,
                    # CNN Encoder/Decoder Config
                    cnn_base_filters: int = 64, # Increased base filters
                    cnn_num_blocks_per_stage: list = [2, 2, 2, 2], # Deeper ResNet structure
                    cnn_feat_dim: int = 512,
                    cnn_groups: int = 8, # GroupNorm groups
                    # GRU Config
                    gru_hidden_size: int = 1024,
                    gru_num_layers: int = 2,
                    gru_dropout: float = 0.1,
                    # Output Config
                    predict_variance: bool = True,
                    predict_residual: bool = True # New flag
                ):
        super().__init__()
        self.predict_variance = predict_variance
        self.predict_residual = predict_residual
        self.noise_img_size = noise_img_size
        self.noise_in_chans = noise_in_chans
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        # --- Components ---
        self.noise_encoder = ResNetCNNEncoder(
            in_chans=noise_in_chans, base_filters=cnn_base_filters,
            num_blocks_per_stage=cnn_num_blocks_per_stage, feat_dim=cnn_feat_dim, groups=cnn_groups
        )

        # GRU input dim = noise feature dim
        gru_input_dim = cnn_feat_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim, hidden_size=gru_hidden_size,
            num_layers=gru_num_layers, batch_first=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0
        )

        # Output Head (CNN Decoder with FiLM)
        # Output channels depend on whether predicting variance
        decoder_out_channels = noise_in_chans * 2 if predict_variance else noise_in_chans
        self.output_decoder = ResNetCNNDecoderFiLM(
            feat_dim=gru_hidden_size, # Input is GRU hidden state
            text_embed_dim=text_embed_dim, # Text embed used for FiLM
            target_chans=decoder_out_channels,
            target_size=noise_img_size,
            base_filters=cnn_base_filters,
            num_blocks_per_stage=cnn_num_blocks_per_stage, # Symmetric to encoder
            groups=cnn_groups
        )

        print(f"Initialized NoiseSequenceRNN_v2:")
        print(f"  - CNN Encoder/Decoder: ResNet Style, BaseFilters={cnn_base_filters}, Blocks={cnn_num_blocks_per_stage}, Groups={cnn_groups}")
        print(f"  - CNN Feature Dim: {cnn_feat_dim}")
        print(f"  - GRU Hidden Size: {gru_hidden_size}, Layers: {gru_num_layers}")
        print(f"  - Text Conditioning: FiLM in Decoder")
        print(f"  - Predict Variance: {predict_variance}")
        print(f"  - Predict Residual: {predict_residual}")


    def forward(self, src_noise_sequence, text_embed, initial_hidden_state=None):
        """
        Forward pass for training (using teacher forcing).

        Args:
            src_noise_sequence (torch.Tensor): [B, SeqLen_in, C, H, W]. Example: [x_T, x'_1, ..., x'_{n-1}]
            text_embed (torch.Tensor): Text embedding [B, D_txt] (pooled).
            initial_hidden_state (torch.Tensor, optional): Initial hidden state for GRU.

        Returns:
            mean_pred_seq or (mean_pred_seq, log_var_pred_seq): Predicted parameters for the *residual* or *full noise*.
            final_hidden_state: Last hidden state of GRU.
        """
        B, T_in, C, H, W = src_noise_sequence.shape
        device = src_noise_sequence.device

        # --- 1. Encode Noise Sequence ---
        noise_flat_batch = src_noise_sequence.view(B * T_in, C, H, W)
        noise_features_flat = self.noise_encoder(noise_flat_batch) # [B * T_in, D_feat]
        noise_features_seq = noise_features_flat.view(B, T_in, -1) # [B, T_in, D_feat]

        # --- 2. Pass through GRU ---
        # GRU input is just the noise features
        gru_output, final_hidden_state = self.gru(noise_features_seq, initial_hidden_state) # [B, T_in, D_hidden]

        # --- 3. Decode GRU Output Sequence using FiLM ---
        gru_output_flat = gru_output.reshape(B * T_in, self.gru_hidden_size)
        # Text embed needs to be expanded for each item in the flattened batch
        text_embed_expanded = text_embed.repeat_interleave(T_in, dim=0) # [B*T_in, D_txt]
        # Decode each step's hidden state, conditioned on text via FiLM
        decoded_params_flat = self.output_decoder(gru_output_flat, text_embed_expanded) # [B * T_in, C_out, H, W]
        # Reshape back to sequence
        output_sequence = decoded_params_flat.view(B, T_in, -1, H, W) # [B, T_in, C_out, H, W]

        # --- 4. Separate Mean and Log Variance ---
        if self.predict_variance:
            # C_out = C * 2
            mean_or_residual_mean_seq, log_var_seq = torch.chunk(output_sequence, 2, dim=2)
            log_var_seq = torch.clamp(log_var_seq, min=-10, max=10)
            return mean_or_residual_mean_seq, log_var_seq, final_hidden_state
        else:
            # C_out = C
            mean_or_residual_mean_seq = output_sequence
            return mean_or_residual_mean_seq, final_hidden_state

    def generate_sequence(self, initial_noise, text_embed, num_steps):
        """ Autoregressive generation for inference. """
        self.eval()
        B, C, H, W = initial_noise.shape
        device = initial_noise.device
        generated_sequence = []
        current_noise_input = initial_noise # x_T or x'_{k-1}
        hidden_state = None

        with torch.no_grad():
            for k in range(num_steps):
                # Prepare input sequence (length 1)
                current_noise_seq = current_noise_input.unsqueeze(1) # [B, 1, C, H, W]

                # Forward pass for one step
                if self.predict_variance:
                    mean_or_residual_mean, log_var, hidden_state = self.forward(current_noise_seq, text_embed, hidden_state)
                    # Remove sequence dimension (dim=1)
                    mean_or_residual_mean = mean_or_residual_mean.squeeze(1) # [B, C, H, W]
                    log_var = log_var.squeeze(1) # [B, C, H, W]
                    # Sample residual or full noise
                    std_dev = torch.exp(0.5 * log_var)
                    sampled_delta_or_full = mean_or_residual_mean + torch.randn_like(mean_or_residual_mean) * std_dev
                else:
                    mean_or_residual_mean, hidden_state = self.forward(current_noise_seq, text_embed, hidden_state)
                    mean_or_residual_mean = mean_or_residual_mean.squeeze(1)
                    sampled_delta_or_full = mean_or_residual_mean # Use mean directly

                # Calculate next noise state
                if self.predict_residual:
                    next_noise = current_noise_input + sampled_delta_or_full # x'_k = x'_{k-1} + Delta_k
                else:
                    next_noise = sampled_delta_or_full # x'_k = mu_k

                generated_sequence.append(next_noise)
                current_noise_input = next_noise # Update for next iteration

        return torch.stack(generated_sequence, dim=1) # [B, N, C, H, W]

