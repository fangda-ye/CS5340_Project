# model/npnet.py
import torch
import torch.nn as nn
import os

# Import components from sibling files
try:
    from .SVDNoiseUnet import SVDNoiseUnet
    from .NoiseTransformer import NoiseTransformer
except ImportError:
    # Fallback for direct script execution (less common)
    from SVDNoiseUnet import SVDNoiseUnet
    from NoiseTransformer import NoiseTransformer

# AdaGroupNorm import
try:
    from diffusers.models.normalization import AdaGroupNorm
except ImportError:
    try:
        from diffusers.models.unet_2d_condition import AdaGroupNorm
    except ImportError:
        raise ImportError("Could not import AdaGroupNorm from diffusers.")

class CrossAttention(nn.Module):
    """ Simple Cross-Attention module. """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        """
        Initializes CrossAttention.

        Args:
            query_dim (int): Dimension of query tensor.
            context_dim (int, optional): Dimension of context tensor. If None, defaults to query_dim.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            dropout (float): Dropout probability.
        """
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        Forward pass for CrossAttention.

        Args:
            x (torch.Tensor): Query tensor (e.g., noise features) [B, C, H, W] or [B, N, D].
            context (torch.Tensor, optional): Context tensor (e.g., text embeddings or text_emb)
                                                [B, M, D_ctx] or needs reshaping. Defaults to None (self-attention).
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor with same shape as x.
        """
        h = self.heads
        # Store original shape
        original_shape = x.shape
        is_spatial = x.dim() == 4

        # Reshape spatial input (B, C, H, W) to sequence (B, N, C) where N = H*W
        if is_spatial:
            b, c, height, width = x.shape
            x = x.view(b, c, height * width).transpose(1, 2) # [B, N, C]

        q = self.to_q(x) # [B, N, inner_dim]

        # Use context if provided, otherwise use x for self-attention
        context = context if context is not None else x

        # Reshape context if it's spatial [B, C_ctx, H, W] -> [B, M, C_ctx]
        if context.dim() == 4:
                b_ctx, c_ctx, h_ctx, w_ctx = context.shape
                context = context.view(b_ctx, c_ctx, h_ctx * w_ctx).transpose(1, 2) # [B, M, C_ctx]

        k = self.to_k(context) # [B, M, inner_dim]
        v = self.to_v(context) # [B, M, inner_dim]

        # Reshape Q, K, V for multi-head attention
        q = q.view(b, -1, h, self.head_dim).transpose(1, 2) # [B, h, N, head_dim]
        k = k.view(b, -1, h, self.head_dim).transpose(1, 2) # [B, h, M, head_dim]
        v = v.view(b, -1, h, self.head_dim).transpose(1, 2) # [B, h, M, head_dim]

        # Compute attention scores
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # [B, h, N, M]

        # Apply mask if provided (e.g., for padding)
        if mask is not None:
            mask = mask.view(b, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = mask[:, None, :].repeat(1, h, 1, 1) # Adjust mask shape [B, h, N, M] ?
            sim.masked_fill_(~mask, max_neg_value)

        # Attention probabilities
        attn = sim.softmax(dim=-1) # [B, h, N, M]

        # Weighted sum of values
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v) # [B, h, N, head_dim]

        # Concatenate heads and project output
        out = out.transpose(1, 2).reshape(b, -1, inner_dim) # [B, N, inner_dim]
        out = self.to_out(out) # [B, N, query_dim]

        # Reshape back to original spatial format if necessary
        if is_spatial:
            out = out.transpose(1, 2).view(b, self.query_dim, height, width) # Use query_dim as C

        return out


class NPNet(nn.Module):
    """
    Noise Prompt Network (NPNet) model - Refactored and Enhanced.

    Combines SVD-based processing and residual prediction using a Transformer
    to transform initial noise into golden noise, conditioned on text prompts.
    Includes options for cross-attention, finetuning, adapter, and dropout.
    """
    def __init__(self, model_id="SDXL", device="cuda", resolution=128,
                    svd_enable_drop=False,
                    nt_enable_adapter=True, nt_enable_finetune=False, nt_enable_dropout=True, # NoiseTransformer options
                    enable_cross_attention=True): # NPNet specific option
        """
        Initializes the NPNet model.

        Args:
            model_id (str): Base diffusion model ID ('SDXL', 'DreamShaper', 'DiT').
            device (str): Device ('cuda' or 'cpu').
            resolution (int): Internal spatial resolution.
            svd_enable_drop (bool): Enable dropout/norm in SVDNoiseUnet.
            nt_enable_adapter (bool): Enable adapter in NoiseTransformer.
            nt_enable_finetune (bool): Enable finetuning in NoiseTransformer.
            nt_enable_dropout (bool): Enable dropout in NoiseTransformer.
            enable_cross_attention (bool): Enable cross-attention between NoiseTransformer output and text_emb.
        """
        super().__init__()
        assert model_id in ["SDXL", "DreamShaper", "DiT"], "Unsupported model_id."

        self.device = device
        self.model_id = model_id
        self.resolution = resolution
        self.enable_cross_attention = enable_cross_attention

        # --- Initialize Components ---
        self.unet_svd = SVDNoiseUnet(
            resolution=resolution,
            enable_drop=svd_enable_drop
        ).to(device).to(torch.float32)

        self.unet_embedding = NoiseTransformer(
            resolution=resolution,
            enable_adapter=nt_enable_adapter,
            enable_finetune=nt_enable_finetune,
            enable_dropout=nt_enable_dropout
        ).to(device).to(torch.float32)

        # Determine text embedding dimension and initialize AdaGroupNorm
        noise_channels = 4 # Assuming latent diffusion noise channels
        self.text_embed_dim_flat = 0 # Placeholder
        if self.model_id == 'DiT':
            self.text_embed_dim_flat = 1024 * 77
            context_dim_for_cross_attn = 1024 # Assuming per-token dim for DiT
        else: # SDXL, DreamShaper
            self.text_embed_dim_flat = 2048 * 77
            context_dim_for_cross_attn = 2048 # Assuming per-token dim for SDXL CLIP

        self.text_embedding = AdaGroupNorm(
            embedding_dim=self.text_embed_dim_flat,
            num_groups=4, # Or make this configurable
            num_channels=noise_channels,
            eps=1e-6
        ).to(device).to(torch.float32)

        # Initialize Cross-Attention if enabled
        self.cross_attn = nn.Identity()
        if self.enable_cross_attention:
                # Query dim should match the output dim of unet_embedding (NoiseTransformer)
                # Context dim should match the dimension of the text embedding used for attention
                # AdaGroupNorm output `text_emb` is [B, C, H, W]. Need to decide context format.
                # Option 1: Use `text_emb` directly (reshaped). Query dim = C, Context dim = C.
                # Option 2: Use original `prompt_embeds` (before flatten). Query dim = C, Context dim = per-token embed dim.
                # Option 2 seems more standard for cross-attention with text.
                query_dim = self.unet_embedding.output_channels # Should be 4
                # Note: CrossAttention expects context [B, M, D_ctx]. We have prompt_embeds [B, SeqLen, EmbDim]
                # And text_emb [B, C, H, W].
                # Let's assume we attend to the spatial text_emb features.
                self.cross_attn = CrossAttention(query_dim=query_dim, context_dim=query_dim).to(device).to(torch.float32)
                print(f"NPNet: Cross-Attention enabled (Query/Context Dim: {query_dim})")


        # Initialize learnable alpha and beta parameters
        self._alpha = nn.Parameter(torch.tensor(0.0, device=self.device))
        self._beta = nn.Parameter(torch.tensor(0.0, device=self.device))

        print(f"Initialized NPNet for {model_id} on {device}.")
        print(f"  - SVDNoiseUnet dropout: {svd_enable_drop}")
        print(f"  - NoiseTransformer adapter: {nt_enable_adapter}")
        print(f"  - NoiseTransformer finetune: {nt_enable_finetune}")
        print(f"  - NoiseTransformer dropout: {nt_enable_dropout}")
        print(f"  - Cross-Attention: {self.enable_cross_attention}")


    def load_pretrained_weights(self, pretrained_path):
        """ Loads weights from a .pth file. """
        if pretrained_path and os.path.isfile(pretrained_path) and pretrained_path.endswith(".pth"):
            try:
                state_dict = torch.load(pretrained_path, map_location=self.device)
                print(f"Loading state dict with keys: {list(state_dict.keys())}") # Debug keys

                # Load sub-module weights
                self.unet_svd.load_state_dict(state_dict["unet_svd"])
                self.unet_embedding.load_state_dict(state_dict["unet_embedding"])

                text_emb_key = "embeeding" if "embeeding" in state_dict else "text_embedding"
                if text_emb_key in state_dict:
                        self.text_embedding.load_state_dict(state_dict[text_emb_key])
                else:
                        print(f"Warning: Key '{text_emb_key}' not found. Text embedding layer not loaded.")

                # Load cross-attention weights if it exists in the state_dict and is enabled
                if self.enable_cross_attention and "cross_attn.to_q.weight" in state_dict: # Check for a specific key
                        # Load weights prefixed with 'cross_attn.'
                        cross_attn_state_dict = {k.replace('cross_attn.', ''): v for k, v in state_dict.items() if k.startswith('cross_attn.')}
                        if cross_attn_state_dict:
                            self.cross_attn.load_state_dict(cross_attn_state_dict)
                            print("Loaded Cross-Attention weights.")
                        else:
                            print("Warning: Cross-Attention enabled but no weights found in state_dict.")
                elif self.enable_cross_attention:
                        print("Warning: Cross-Attention enabled but no weights found in state_dict.")


                # Load alpha and beta
                if "alpha" in state_dict:
                    self._alpha.data.copy_(state_dict["alpha"].to(self.device).reshape(self._alpha.data.shape))
                elif "_alpha" in state_dict: # Check for private name convention
                        self._alpha.data.copy_(state_dict["_alpha"].to(self.device).reshape(self._alpha.data.shape))
                else:
                    print("Warning: 'alpha'/'_alpha' not found in state_dict.")

                if "beta" in state_dict:
                        self._beta.data.copy_(state_dict["beta"].to(self.device).reshape(self._beta.data.shape))
                elif "_beta" in state_dict:
                        self._beta.data.copy_(state_dict["_beta"].to(self.device).reshape(self._beta.data.shape))
                else:
                        print("Warning: 'beta'/'_beta' not found in state_dict.")

                print(f"Successfully loaded pretrained NPNet weights from {pretrained_path}")
            except FileNotFoundError:
                print(f"Error: Pretrained weights file not found at {pretrained_path}")
            except KeyError as e:
                print(f"Error: Key error loading state dict ({e}). Check keys in the .pth file.")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        else:
            print(f"Warning: No valid pretrained weights path provided or file not found: {pretrained_path}. Model uses initial weights.")


    def forward(self, initial_noise, prompt_embeds):
        """
        Forward pass of the NPNet model.

        Args:
            initial_noise (torch.Tensor): Initial random noise [B, C, H, W].
            prompt_embeds (torch.Tensor): Original text embeddings [B, SeqLen, EmbDim].
                                            Used for AdaGroupNorm (flattened) and potentially Cross-Attention.

        Returns:
            torch.Tensor: Predicted golden noise [B, C, H, W].
        """
        initial_noise = initial_noise.to(self.device).to(torch.float32)
        prompt_embeds = prompt_embeds.to(self.device).to(torch.float32)

        # 1. Flatten prompt embeddings for AdaGroupNorm
        batch_size = prompt_embeds.shape[0]
        prompt_embeds_flat = prompt_embeds.view(batch_size, -1)

        # 2. Calculate text_emb using AdaGroupNorm
        text_emb = self.text_embedding(initial_noise, prompt_embeds_flat) # Shape: [B, C, H, W]

        # 3. Process through NoiseTransformer (Residual Branch)
        encoder_hidden_states_embedding = initial_noise + text_emb
        golden_embedding = self.unet_embedding(encoder_hidden_states_embedding) # Shape: [B, C_out, H, W]

        # 4. Apply Cross-Attention if enabled
        # Attends golden_embedding (Query) to text_emb (Context)
        if self.enable_cross_attention:
                # CrossAttention expects Query [B, N, D_q] and Context [B, M, D_c]
                # Here, Query is golden_embedding [B, C, H, W]
                # Context is text_emb [B, C, H, W]
                # The CrossAttention class handles reshaping internally.
                golden_embedding = self.cross_attn(golden_embedding, context=text_emb)

        # 5. Process through SVDNoiseUnet (SVD Branch)
        svd_output = self.unet_svd(initial_noise)

        # 6. Combine outputs
        alpha_coeff = (2 * torch.sigmoid(self._alpha) - 1)
        beta_coeff = self._beta

        golden_noise = svd_output + alpha_coeff * text_emb + beta_coeff * golden_embedding

        return golden_noise # Return in float32
