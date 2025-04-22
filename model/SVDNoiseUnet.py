# model/SVDNoiseUnet.py
import torch
import torch.nn as nn
import einops
from torch.nn import functional as F
import os # Added for isfile check

# Import Attention from the separate Attention.py file
try:
    from .Attention import Attention, DropPath # Import DropPath as well if used
except ImportError:
    from Attention import Attention, DropPath # Fallback for direct execution

__all__ = ['SVDNoiseUnet', 'SVDNoiseUnet_Concise']

class SVDNoiseUnet(nn.Module):
    """
    Noise processing module based on Singular Value Decomposition (SVD).
    Predicts modified singular values to reconstruct the noise.
    Includes an option for dropout and more complex MLPs.
    """
    def __init__(self, in_channels=4, out_channels=4, resolution=128, enable_drop=True):
        """
        Initializes the SVDNoiseUnet module.

        Args:
            in_channels (int): Number of input channels. Defaults to 4.
            out_channels (int): Number of output channels. Defaults to 4.
            resolution (int): Spatial resolution for calculating internal dimensions. Defaults to 128.
            enable_drop (bool): Whether to use LayerNorm, GELU, Dropout, DropPath
                                instead of simpler ReLU MLPs. Defaults to True.
        """
        super(SVDNoiseUnet, self).__init__()

        self.enable_drop = enable_drop
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate internal feature dimension based on rearrangement
        # Example: (B, 4, 128, 128) -> rearrange(a=2, c=2) -> (B, 256, 256)
        # _flat_dim becomes 256
        _flat_dim = resolution * in_channels // 2
        # print(f"SVDNoiseUnet: Internal flat dimension = {_flat_dim}")

        # Define network components based on enable_drop flag
        if self.enable_drop:
            # More complex version with Norm, GELU, Dropout, DropPath
            mlp_hidden_dim = 256 # Intermediate hidden dimension for MLPs
            self.mlp1 = nn.Sequential(
                nn.Linear(_flat_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(mlp_hidden_dim, _flat_dim), # Output dim matches flat_dim
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(_flat_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(mlp_hidden_dim, _flat_dim),
            )
            self.mlp3 = nn.Sequential( # Processes singular values 's'
                nn.Linear(_flat_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(mlp_hidden_dim, _flat_dim),
            )
            # Attention block applied to combined features
            self.attention = nn.Sequential(
                Attention(dim=_flat_dim, num_heads=8, qkv_bias=True, attn_drop=0.1, proj_drop=0.1),
                # DropPath typically applied after residual connection, but here applied after attention output
                DropPath(0.1)
            )
            # MLP applied after attention and aggregation
            mlp4_hidden_dim = 1024
            self.mlp4 = nn.Sequential(
                nn.Linear(_flat_dim, mlp4_hidden_dim),
                nn.LayerNorm(mlp4_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(mlp4_hidden_dim, _flat_dim),
            )
        else:
            # Simpler version with ReLU and no dropout (closer to original paper snippet)
            mlp_hidden_dim = 64
            self.mlp1 = nn.Sequential(
                nn.Linear(_flat_dim, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_dim, _flat_dim),
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(_flat_dim, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_dim, _flat_dim),
            )
            self.mlp3 = nn.Sequential( # Processes singular values 's'
                nn.Linear(_flat_dim, _flat_dim), # Direct mapping or simple MLP
            )
            self.attention = Attention(dim=_flat_dim, num_heads=8, qkv_bias=True) # Basic attention
            mlp4_hidden_dim = 1024
            self.mlp4 = nn.Sequential(
                nn.Linear(_flat_dim, mlp4_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp4_hidden_dim, _flat_dim),
            )

        # BatchNorm layer (Usage remains questionable after attention.mean(1))
        # self.bn = nn.BatchNorm2d(out_channels) # Input should be 4D: B, C, H, W
        # If normalization is needed after attention.mean(1) (output is B, _flat_dim),
        # LayerNorm would be more appropriate:
        # self.post_attn_norm = nn.LayerNorm(_flat_dim)

    def forward(self, x):
        """
        Forward pass for SVDNoiseUnet.

        Args:
            x (torch.Tensor): Input noise tensor [B, C_in, H, W].

        Returns:
            torch.Tensor: Reconstructed noise tensor [B, C_out, H, W].
        """
        b, c_in, h, w = x.shape
        assert c_in == self.in_channels, f"Input channel mismatch: got {c_in}, expected {self.in_channels}"

        # --- SVD Processing ---
        # 1. Rearrange input for SVD: [B, C, H, W] -> [B, flat_H, flat_W]
        # Example: [B, 4, 128, 128] -> [B, 256, 256] (assuming a=2, c=2)
        a = 2 # Factor for height dimension
        c_factor = c_in // a # Factor for channel dimension
        if c_in % a != 0:
                raise ValueError(f"Input channels ({c_in}) must be divisible by rearrangement factor a ({a})")
        x_rearranged = einops.rearrange(x, "b (a c) h w -> b (a h) (c w)", a=a, c=c_factor)
        current_flat_dim = x_rearranged.shape[-1] # Actual dimension after rearrangement

        # 2. Perform Singular Value Decomposition
        try:
            # Use Vh=True to get V transpose directly, matching common notation U @ diag(s) @ Vh
            U, s, Vh = torch.linalg.svd(x_rearranged, full_matrices=False) # full_matrices=False is more efficient
            # U: [B, flat_H, K], s: [B, K], Vh: [B, K, flat_W], where K=min(flat_H, flat_W)
            # For square matrices K = flat_dim
        except torch.linalg.LinAlgError as e:
                print(f"SVD failed: {e}. Returning input tensor.")
                # Return input unmodified if SVD fails
                return x

        # Ensure dimensions match expected _flat_dim if needed for MLPs
        # This might require padding/truncation if K != _flat_dim, but usually they match for square rearrange
        if s.shape[-1] != current_flat_dim:
                print(f"Warning: SVD dimension {s.shape[-1]} does not match expected flat dimension {current_flat_dim}. Check rearrangement.")
                # Handle mismatch if necessary (e.g., padding s, U, Vh)

        # 3. Process SVD components through MLPs
        # Note: mlp1 expects input dim _flat_dim, but U is [B, flat_H, K].
        # The original code applied mlp1 to U_T ([B, K, flat_H]). This assumes K=flat_H=_flat_dim.
        # Similarly for mlp2 and V (from Vh).
        # Let's assume K = flat_dim for simplicity based on square rearrangement.
        U_T = U.permute(0, 2, 1)  # [B, K, flat_H] -> [B, _flat_dim, _flat_dim] if square
        V = Vh.mH # Conjugate transpose to get V: [B, K, flat_W] -> [B, flat_W, K] -> [B, _flat_dim, _flat_dim]

        out_u = self.mlp1(U_T) # Input [B, dim, dim], Output [B, dim, dim]
        out_v = self.mlp2(V)   # Input [B, dim, dim], Output [B, dim, dim]
        out_s = self.mlp3(s)   # Input [B, dim], Output [B, dim]
        out_s_unsqueezed = out_s.unsqueeze(1) # Add dim for broadcasting: [B, 1, dim]

        # Combine processed components (element-wise add with broadcasting)
        combined = out_u + out_v + out_s_unsqueezed # Shape: [B, dim, dim]

        # --- Attention and Refinement ---
        # 4. Apply Attention
        # Input: [B, dim, dim]. Attention expects [B, SeqLen, FeatureDim].
        # Treat one dim as SeqLen, the other as FeatureDim. Let's use dim=SeqLen, dim=FeatureDim.
        attn_output = self.attention(combined) # Shape: [B, dim, dim]

        # 5. Aggregate features (e.g., mean pooling)
        # Original code used mean(1), averaging over the SeqLen dimension.
        aggregated_features = attn_output.mean(dim=1) # Shape: [B, dim]

        # Optional: Apply normalization after aggregation
        # if hasattr(self, 'post_attn_norm'):
        #     aggregated_features = self.post_attn_norm(aggregated_features)

        # 6. Refine singular values using mlp4 and residual connection
        refined_s = self.mlp4(aggregated_features) + s # Shape: [B, dim]

        # --- Reconstruction ---
        # 7. Ensure predicted singular values are non-negative (optional but recommended)
        refined_s = F.relu(refined_s)

        # 8. Reconstruct the matrix using original U, Vh and refined s
        # U: [B, dim, dim], diag_embed(refined_s): [B, dim, dim], Vh: [B, dim, dim]
        pred = U @ torch.diag_embed(refined_s) @ Vh # Shape: [B, dim, dim]

        # 9. Rearrange back to original image tensor shape
        # [B, dim, dim] -> [B, (a*c), H, W]
        pred_rearranged = einops.rearrange(pred, "b (a h) (c w) -> b (a c) h w", a=a, c=c_factor, h=h, w=w)

        # Ensure output channels match
        if pred_rearranged.shape[1] != self.out_channels:
                # This shouldn't happen if in_channels == out_channels and rearrangement is symmetric
                print(f"Warning: Output channels mismatch. Got {pred_rearranged.shape[1]}, expected {self.out_channels}. Check rearrangement factors.")
                # May need a final conv layer if channel change is intended/unavoidable
                # pred_rearranged = self.final_conv(pred_rearranged)

        return pred_rearranged


class SVDNoiseUnet_Concise(nn.Module):
    """ Placeholder for a concise version of SVDNoiseUnet. """
    def __init__(self, in_channels=4, out_channels=4, resolution=128):
        super(SVDNoiseUnet_Concise, self).__init__()
        print("Warning: SVDNoiseUnet_Concise is not implemented.")
        # Add simplified implementation here if needed
        pass

    def forward(self, x):
        # Placeholder: just return input
        print("Warning: SVDNoiseUnet_Concise forward pass is not implemented. Returning input.")
        return x

# Example Usage (if run directly)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test with dropout enabled
    model_drop = SVDNoiseUnet(resolution=128, enable_drop=True).to(device)
    model_drop.eval()
    # Test without dropout
    model_no_drop = SVDNoiseUnet(resolution=128, enable_drop=False).to(device)
    model_no_drop.eval()

    # Create dummy input (like SD latent noise)
    dummy_input = torch.randn(2, 4, 128, 128).to(device) # Batch=2, Channels=4, H=128, W=128

    print("\nTesting SVDNoiseUnet (enable_drop=True)...")
    print("Input Shape:", dummy_input.shape)
    with torch.no_grad():
        output_drop = model_drop(dummy_input)
    print("Output Shape:", output_drop.shape)
    # Expected output shape: (2, 4, 128, 128)

    print("\nTesting SVDNoiseUnet (enable_drop=False)...")
    print("Input Shape:", dummy_input.shape)
    with torch.no_grad():
        output_no_drop = model_no_drop(dummy_input)
    print("Output Shape:", output_no_drop.shape)
    # Expected output shape: (2, 4, 128, 128)
