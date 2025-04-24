# model/rnn_seq_model_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# --- Helper Modules (FiLMLayer, ResBlock, ResNetCNNEncoder) ---
# (Copied from previous rnn_seq_model_v2.py - Ensure these are correct)
class FiLMLayer(nn.Module):
    """Applies Feature-wise Linear Modulation"""
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        # Ensure gamma and beta match x's channel dimension
        if gamma.shape[1] != x.shape[1]:
                raise ValueError(f"FiLM gamma/beta channel size ({gamma.shape[1]}) doesn't match input feature channel size ({x.shape[1]})")
        return gamma * x + beta

class ResBlock(nn.Module):
    """ Simple Residual Block with GroupNorm and SiLU activation """
    def __init__(self, in_channels, out_channels, groups=8, dropout=0.1):
        super().__init__()
        # Ensure groups is at least 1 and divides in_channels if possible
        groups_in = min(groups, in_channels) if in_channels >= groups else 1
        while in_channels % groups_in != 0 and groups_in > 1: groups_in //= 2 # Find suitable group size

        groups_out = min(groups, out_channels) if out_channels >= groups else 1
        while out_channels % groups_out != 0 and groups_out > 1: groups_out //= 2

        self.norm1 = nn.GroupNorm(num_groups=groups_in, num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.norm1(x); x = self.act1(x); x = self.conv1(x)
        x = self.norm2(x); x = self.act2(x); x = self.dropout(x)
        x = self.conv2(x)
        return x + residual

class ResNetCNNEncoder(nn.Module):
    """ ResNet style CNN to encode noise [B, C, H, W] -> [B, D_feat] """
    def __init__(self, in_chans=4, base_filters=32, num_blocks_per_stage=[2, 2, 2], feat_dim=512, groups=8):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_chans, base_filters, kernel_size=3, padding=1)
        current_chans = base_filters
        self.stages = nn.ModuleList()
        for i, num_blocks in enumerate(num_blocks_per_stage):
            stage_layers = []; out_chans = base_filters * (2**i)
            if i > 0: stage_layers.append(nn.Conv2d(current_chans, out_chans, kernel_size=3, stride=2, padding=1))
            else: stage_layers.append(nn.Conv2d(current_chans, out_chans, kernel_size=3, stride=1, padding=1))
            current_chans = out_chans
            for _ in range(num_blocks): stage_layers.append(ResBlock(current_chans, current_chans, groups=groups))
            self.stages.append(nn.Sequential(*stage_layers))

        groups_final = min(groups, current_chans) if current_chans >= groups else 1
        while current_chans % groups_final != 0 and groups_final > 1: groups_final //= 2
        self.final_norm = nn.GroupNorm(num_groups=groups_final, num_channels=current_chans)
        self.final_act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(current_chans, feat_dim)

    def forward(self, x):
        x = self.initial_conv(x)
        for stage in self.stages: x = stage(x)
        x = self.final_norm(x); x = self.final_act(x)
        x = self.pool(x); x = self.flatten(x); x = self.fc(x)
        return x

class ResNetCNNDecoderFiLM(nn.Module):
    """ ResNet style CNN Decoder with FiLM conditioning """
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
        self.film_layer = FiLMLayer()

        current_chans = initial_chans
        for i in range(num_stages - 1, -1, -1):
            stage_res_blocks = []; film_mod_chans = current_chans # Channels to modulate
            for _ in range(num_blocks_per_stage[i]):
                stage_res_blocks.append(ResBlock(current_chans, current_chans, groups=groups))
            self.res_blocks.append(nn.Sequential(*stage_res_blocks))
            film_out_dim = film_mod_chans * 2
            self.film_generators.append(nn.Linear(text_embed_dim, film_out_dim))

            out_chans = base_filters * (2**(i-1)) if i > 0 else base_filters
            if i > 0:
                    self.upsamplers.append(nn.ConvTranspose2d(current_chans, out_chans, kernel_size=4, stride=2, padding=1))
                    current_chans = out_chans
            else:
                    self.upsamplers.append(nn.Identity())
        self.final_conv = nn.Conv2d(current_chans, target_chans, kernel_size=3, padding=1)

    def forward(self, x, text_embed):
        x = self.initial_fc(x); x = x.view(-1, *self.initial_reshape_size)
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
            film_params = self.film_generators[i](text_embed)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            x = self.film_layer(x, gamma, beta)
            x = self.upsamplers[i](x)
        x = self.final_conv(x)
        if x.shape[-2:] != (self.target_size, self.target_size):
                x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        return x

# --- Final RNN Sequence Model (V3) ---
class NoiseSequenceRNN_v3(nn.Module):
    """
    RNN (GRU) model predicting full next noise state distribution,
    with text conditioning in GRU input and FiLM in decoder. Includes KL loss support.
    """
    def __init__(self,
                    text_embed_dim: int, noise_img_size: int = 128, noise_in_chans: int = 4,
                    cnn_base_filters: int = 64, cnn_num_blocks_per_stage: list = [2, 2, 2, 2],
                    cnn_feat_dim: int = 512, cnn_groups: int = 8,
                    gru_hidden_size: int = 1024, gru_num_layers: int = 2, gru_dropout: float = 0.1,
                    predict_variance: bool = True):
        super().__init__()
        self.predict_variance = predict_variance
        self.noise_img_size = noise_img_size
        self.noise_in_chans = noise_in_chans
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        # --- Components ---
        self.noise_encoder = ResNetCNNEncoder(
            in_chans=noise_in_chans, base_filters=cnn_base_filters,
            num_blocks_per_stage=cnn_num_blocks_per_stage, feat_dim=cnn_feat_dim, groups=cnn_groups
        )

        # Optional: Project text embedding to match desired input size for GRU concat
        # Let's make GRU input = cnn_feat_dim + projected_text_dim
        projected_text_dim = cnn_feat_dim # Example: project text to same dim as noise feature
        self.text_projection = nn.Linear(text_embed_dim, projected_text_dim)
        gru_input_dim = cnn_feat_dim + projected_text_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim, hidden_size=gru_hidden_size,
            num_layers=gru_num_layers, batch_first=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0
        )

        # Output Head (CNN Decoder with FiLM)
        decoder_out_channels = noise_in_chans * 2 if predict_variance else noise_in_chans
        self.output_decoder = ResNetCNNDecoderFiLM(
            feat_dim=gru_hidden_size, text_embed_dim=text_embed_dim, # Use original text embed dim for FiLM
            target_chans=decoder_out_channels, target_size=noise_img_size,
            base_filters=cnn_base_filters, num_blocks_per_stage=cnn_num_blocks_per_stage, groups=cnn_groups
        )

        print(f"Initialized NoiseSequenceRNN_v3:")
        print(f"  - CNN Encoder/Decoder: ResNet Style, BaseFilters={cnn_base_filters}, Blocks={cnn_num_blocks_per_stage}, Groups={cnn_groups}")
        print(f"  - CNN Feature Dim: {cnn_feat_dim}")
        print(f"  - GRU Input Dim: {gru_input_dim} (NoiseFeat={cnn_feat_dim} + TextProj={projected_text_dim})")
        print(f"  - GRU Hidden Size: {gru_hidden_size}, Layers: {gru_num_layers}")
        print(f"  - Text Conditioning: GRU Input Concat + FiLM in Decoder")
        print(f"  - Predict Variance: {predict_variance}")
        print(f"  - Predicts Full State (Not Residual)")

    def forward(self, src_noise_sequence, text_embed, initial_hidden_state=None):
        """
        Forward pass for training (using teacher forcing). Predicts parameters for x'_k.

        Args:
            src_noise_sequence (torch.Tensor): [B, SeqLen_in, C, H, W]. Example: [x_T, x'_1, ..., x'_{n-1}]
            text_embed (torch.Tensor): Text embedding [B, D_txt] (pooled).
            initial_hidden_state (torch.Tensor, optional): Initial hidden state for GRU.

        Returns:
            mean_pred_seq or (mean_pred_seq, log_var_pred_seq): Predicted parameters for the *full noise state* x'_k.
            final_hidden_state: Last hidden state of GRU.
        """
        B, T_in, C, H, W = src_noise_sequence.shape
        device = src_noise_sequence.device

        # --- 1. Encode Noise Sequence ---
        noise_flat_batch = src_noise_sequence.view(B * T_in, C, H, W)
        noise_features_flat = self.noise_encoder(noise_flat_batch) # [B * T_in, D_feat]
        noise_features_seq = noise_features_flat.view(B, T_in, -1) # [B, T_in, D_feat]

        # --- 2. Prepare GRU Input with Text ---
        projected_text = self.text_projection(text_embed) # [B, D_proj]
        projected_text_expanded = projected_text.unsqueeze(1).expand(-1, T_in, -1) # [B, T_in, D_proj]
        gru_input_seq = torch.cat([noise_features_seq, projected_text_expanded], dim=-1) # [B, T_in, D_feat + D_proj]

        # --- 3. Pass through GRU ---
        gru_output, final_hidden_state = self.gru(gru_input_seq, initial_hidden_state) # [B, T_in, D_hidden]

        # --- 4. Decode GRU Output Sequence using FiLM ---
        gru_output_flat = gru_output.reshape(B * T_in, self.gru_hidden_size)
        # FiLM uses the original (unprojected) text embedding
        text_embed_expanded_film = text_embed.repeat_interleave(T_in, dim=0) # [B*T_in, D_txt_orig]
        decoded_params_flat = self.output_decoder(gru_output_flat, text_embed_expanded_film) # [B * T_in, C_out, H, W]
        output_sequence = decoded_params_flat.view(B, T_in, -1, H, W) # [B, T_in, C_out, H, W]

        # --- 5. Separate Mean and Log Variance ---
        if self.predict_variance:
            mean_seq, log_var_seq = torch.chunk(output_sequence, 2, dim=2)
            log_var_seq = torch.clamp(log_var_seq, min=-10, max=10) # Stability clamp
            return mean_seq, log_var_seq, final_hidden_state
        else:
            mean_seq = output_sequence
            return mean_seq, final_hidden_state

    def generate_sequence(self, initial_noise, text_embed, num_steps):
        """ Autoregressive generation for inference. Predicts full state x'_k. """
        self.eval()
        B, C, H, W = initial_noise.shape
        device = initial_noise.device
        generated_sequence = []
        current_noise_input = initial_noise # x_T
        hidden_state = None
        # Project text embed once for GRU input
        projected_text = self.text_projection(text_embed) # [B, D_proj]

        with torch.no_grad():
            for k in range(num_steps):
                # --- Encode current noise ---
                noise_feat = self.noise_encoder(current_noise_input) # [B, D_feat]

                # --- Prepare GRU input ---
                gru_input = torch.cat([noise_feat, projected_text], dim=-1).unsqueeze(1) # [B, 1, D_feat + D_proj]

                # --- GRU Step ---
                gru_output, hidden_state = self.gru(gru_input, hidden_state) # [B, 1, D_hidden]

                # --- Decode Step ---
                gru_output_flat = gru_output.squeeze(1) # [B, D_hidden]
                # FiLM uses original text embed
                decoded_params = self.output_decoder(gru_output_flat, text_embed) # [B, C_out, H, W]

                # --- Sample Next Noise ---
                if self.predict_variance:
                    mean_pred, log_var_pred = torch.chunk(decoded_params, 2, dim=1) # Split channels
                    log_var_pred = torch.clamp(log_var_pred, min=-10, max=10)
                    std_dev = torch.exp(0.5 * log_var_pred)
                    next_noise = mean_pred + torch.randn_like(mean_pred) * std_dev
                else:
                    mean_pred = decoded_params
                    next_noise = mean_pred # Use mean directly

                generated_sequence.append(next_noise)
                current_noise_input = next_noise # Update for next iteration

        return torch.stack(generated_sequence, dim=1) # [B, N, C, H, W]

