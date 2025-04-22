# model/Attention.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.jit import Final
import os # Added for isfile check in load_pretrained_weights

# Assuming timm is installed and needed for use_fused_attn
try:
    from timm.layers import use_fused_attn, DropPath
except ImportError:
    print("Warning: timm library not found or use_fused_attn/DropPath not available.")
    # Provide a fallback or raise an error if fused attention is critical
    def use_fused_attn(): return False # Simple fallback
    # Dummy DropPath if timm not available
    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            return x

__all__ = ['Attention'] # Define what is exported from this module

class Attention(nn.Module):
    """
    Standard Multi-Head Self-Attention module.

    Typically used within Transformer-based architectures or similar models
    requiring sequence self-attention.
    """
    fused_attn: Final[bool] # Type hint for TorchScript

    def __init__(
            self,
            dim: int,               # Input feature dimension
            num_heads: int = 8,     # Number of attention heads
            qkv_bias: bool = False, # Whether to add bias to Q, K, V projections
            qk_norm: bool = False,  # Whether to normalize Q and K before attention
            attn_drop: float = 0.,  # Dropout rate for attention weights
            proj_drop: float = 0.,  # Dropout rate for the final output projection
            norm_layer: nn.Module = nn.LayerNorm, # Normalization layer to use (if qk_norm is True)
    ) -> None:
        """ Initializes the Attention module. """
        super().__init__()
        # Ensure feature dimension is divisible by the number of heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads # Dimension per attention head
        self.scale = self.head_dim ** -0.5 # Scaling factor for dot products
        self.fused_attn = use_fused_attn() # Check if hardware supports fused attention

        # Linear layer for combined Query, Key, Value projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Optional normalization layers for Query and Key
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) # Final linear projection layer
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Attention module.

        Args:
            x: Input tensor of shape (Batch, SequenceLength, FeatureDim).

        Returns:
            Output tensor of the same shape (Batch, SequenceLength, FeatureDim).
        """
        B, N, C = x.shape # Batch size, Sequence length, Channels (feature dimension)

        # 1. Project to Q, K, V and reshape for multi-head attention
        # qkv shape: (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim)
        # permute to: (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Separate Q, K, V tensors (each shape: B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # 2. Apply optional normalization to Q and K
        q, k = self.q_norm(q), self.k_norm(k)

        # 3. Compute attention scores
        if self.fused_attn:
            # Use efficient fused scaled dot-product attention if available
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                # attn_mask=None, # Optional attention mask
                # is_causal=False # Set to True for causal attention (e.g., decoders)
            )
        else:
            # Manual implementation of scaled dot-product attention
            q = q * self.scale # Scale query
            # Attention scores: (B, num_heads, N, N)
            attn = (q @ k.transpose(-2, -1)) # Matrix multiply Q with K transpose
            attn = attn.softmax(dim=-1) # Apply softmax to get attention weights
            attn = self.attn_drop(attn) # Apply dropout to attention weights
            # Weighted sum of values: (B, num_heads, N, head_dim)
            x = (attn @ v) # Matrix multiply attention weights with V

        # 4. Reshape and project output
        # Transpose and reshape: (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x) # Apply final linear projection
        x = self.proj_drop(x) # Apply final dropout

        return x

# Example Usage (if run directly)
if __name__ == '__main__':
    # Example of how to use the Attention module
    input_dim = 256
    seq_len = 50
    batch_size = 4

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # Instantiate the Attention module
    attention_layer = Attention(dim=input_dim, num_heads=8, qkv_bias=True)

    # Pass the input through the layer
    output = attention_layer(dummy_input)

    print("Input Shape:", dummy_input.shape)
    print("Output Shape:", output.shape)
    # Expected output shape: (4, 50, 256)
