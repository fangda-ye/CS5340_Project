# model/NoiseTransformer.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import os # Added for isfile check

try:
    from timm import create_model
except ImportError:
    raise ImportError("timm library is required for NoiseTransformer. Please install it: pip install timm")

__all__ = ['NoiseTransformer']

class NoiseTransformer(nn.Module):
    """
    Noise processing module using a Swin Transformer backbone.
    Includes options for finetuning, adapter, and dropout.
    """
    def __init__(self, resolution=128, input_channels=4, output_channels=4,
                    enable_finetune=False, enable_adapter=True, enable_dropout=True):
        """
        Initializes the NoiseTransformer module.

        Args:
            resolution (int): Target output resolution after downsampling Swin features. Defaults to 128.
            input_channels (int): Number of input channels (e.g., 4 for SD latents). Defaults to 4.
            output_channels (int): Number of output channels. Defaults to 4.
            enable_finetune (bool): Whether to unfreeze and finetune the last two stages of Swin. Defaults to False.
            enable_adapter (bool): Whether to insert a small convolutional adapter before Swin. Defaults to True.
            enable_dropout (bool): Whether to apply Dropout2d after Swin features. Defaults to True.
        """
        super().__init__()
        self.resolution = resolution
        self.target_vit_res = 224 # Swin Transformer input resolution
        self.input_channels = input_channels
        self.output_channels = output_channels

        # --- Layers ---
        # Interpolation functions
        self.upsample = lambda x: F.interpolate(x, size=(self.target_vit_res, self.target_vit_res), mode='bilinear', align_corners=False)
        self.downsample = lambda x: F.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)

        # Convolution to map input channels to 3 for Swin
        self.downconv = nn.Conv2d(self.input_channels, 3, kernel_size=1, stride=1, padding=0)

        # Load Swin Transformer
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=True)
        swin_out_dim = self.swin.num_features # Output feature dimension (e.g., 768)

        # Convolution to map Swin output features back to desired output channels
        # Note: The original upconv(7, 4, ...) seemed incorrect based on Swin output dim.
        # This layer processes the features *after* Swin and *before* final downsampling.
        # It should map swin_out_dim to something, potentially output_channels directly or an intermediate dim.
        # Let's map swin_out_dim -> output_channels.
        self.feature_conv = nn.Conv2d(swin_out_dim, self.output_channels, kernel_size=1, stride=1, padding=0)
        # print(f"NoiseTransformer: Swin output dim = {swin_out_dim}, Feature conv maps to {self.output_channels}")

        # --- Optional Components ---
        self.enable_finetune = enable_finetune
        self.enable_adapter = enable_adapter
        self.enable_dropout = enable_dropout

        # Adapter module (if enabled)
        self.adapter = nn.Identity() # Default to identity if not enabled
        if self.enable_adapter:
            self.adapter = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1), # Operates on 3-channel input to Swin
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=1) # Output should remain 3 channels for Swin
            )

        # Dropout module (if enabled)
        self.dropout = nn.Identity() # Default to identity if not enabled
        if self.enable_dropout:
            # Dropout applied *after* processing Swin features
            self.dropout = nn.Dropout2d(0.2) # Use Dropout2d for spatial features

        # --- Parameter Freezing/Finetuning ---
        # Freeze all Swin parameters initially
        for param in self.swin.parameters():
            param.requires_grad = False

        # Unfreeze last two stages if finetuning is enabled
        if self.enable_finetune:
            if hasattr(self.swin, 'layers') and len(self.swin.layers) >= 2:
                for stage_idx in [-1, -2]: # Unfreeze last two stages
                        for param in self.swin.layers[stage_idx].parameters():
                            param.requires_grad = True
                print("NoiseTransformer: Finetuning enabled for last 2 Swin stages.")
            else:
                print("Warning: Could not enable finetuning. Swin model structure might be different than expected.")


    def forward(self, x):
        """
        Forward pass for NoiseTransformer.

        Args:
            x (torch.Tensor): Input noise tensor [B, C_in, H_in, W_in].
                                Note: The 'residual' flag from the original user code
                                seemed unused or incorrectly implemented, so it's removed here.
                                If residual connection is needed, it should likely happen
                                in the NPNet class combining this output.

        Returns:
            torch.Tensor: Processed noise tensor [B, C_out, H_out, W_out] (H_out=W_out=resolution).
        """
        # --- Input Processing ---
        # 1. Upsample to Swin's expected size (224x224)
        x_upsampled = self.upsample(x)
        # 2. Convert input channels to 3 for Swin
        x_3channel = self.downconv(x_upsampled)
        # 3. Apply adapter if enabled
        x_adapted = self.adapter(x_3channel)

        # --- Swin Transformer Feature Extraction ---
        # Input shape: [B, 3, 224, 224]
        # Output shape: depends on Swin variant, e.g., [B, N, D] like [B, 49, 768] for tiny
        features = self.swin.forward_features(x_adapted)
        # print("Swin features shape:", features.shape) # Debug

        # --- Output Processing ---
        # Need to reshape Swin output back to spatial format [B, D, H_feat, W_feat]
        B, N, D = features.shape
        H_feat = W_feat = int(N**0.5) # Assuming square grid of patches
        features_spatial = features.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)
        # print("Swin features spatial shape:", features_spatial.shape) # Debug

        # 5. Map features to output channels
        # Input: [B, D, H_feat, W_feat], Output: [B, C_out, H_feat, W_feat]
        x_processed = self.feature_conv(features_spatial)
        # print("After feature_conv shape:", x_processed.shape) # Debug

        # 6. Downsample to target resolution
        # Input: [B, C_out, H_feat, W_feat], Output: [B, C_out, resolution, resolution]
        x_downsampled = self.downsample(x_processed)
        # print("After downsample shape:", x_downsampled.shape) # Debug

        # 7. Apply dropout if enabled
        output = self.dropout(x_downsampled)

        return output

# Example Usage (if run directly)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test with default settings
    model = NoiseTransformer(resolution=128, enable_adapter=True, enable_dropout=True).to(device)
    model.eval() # Set to eval mode for testing

    # Create dummy input (like SD latent noise)
    dummy_input = torch.randn(2, 4, 128, 128).to(device) # Batch=2, Channels=4, H=128, W=128

    print("\nTesting NoiseTransformer...")
    print("Input Shape:", dummy_input.shape)

    with torch.no_grad():
        output = model(dummy_input)

    print("Output Shape:", output.shape)
    # Expected output shape: (2, 4, 128, 128)

    # Test with finetuning enabled
    # model_ft = NoiseTransformer(resolution=64, enable_finetune=True, enable_adapter=False).to(device)
    # print("\nModel with Finetuning Enabled:")
    # for name, param in model_ft.named_parameters():
    #      if 'swin' in name:
    #           print(f"  Param: {name}, Requires Grad: {param.requires_grad}")
