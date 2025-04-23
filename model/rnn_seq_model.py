# model/rnn_seq_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- Helper Modules ---

class SimpleCNNEncoder(nn.Module):
    """ Encodes a noise tensor [B, C, H, W] into a feature vector [B, D_feat] """
    def __init__(self, in_chans=4, base_filters=32, num_layers=3, feat_dim=512, activation=nn.ReLU):
        super().__init__()
        layers = []
        current_chans = in_chans
        for i in range(num_layers):
            out_chans = base_filters * (2**i)
            layers.append(nn.Conv2d(current_chans, out_chans, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_chans)) # Use BatchNorm or GroupNorm
            layers.append(activation())
            current_chans = out_chans

        layers.append(nn.AdaptiveAvgPool2d((1, 1))) # Global average pooling
        layers.append(nn.Flatten())
        layers.append(nn.Linear(current_chans, feat_dim)) # Project to final feature dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class SimpleCNNDecoder(nn.Module):
    """ Decodes a feature vector [B, D_feat] back into noise parameters [B, C_out, H, W] """
    def __init__(self, feat_dim=512, target_chans=4, target_size=128, base_filters=32, num_layers=3, activation=nn.ReLU):
        super().__init__()
        self.target_chans = target_chans
        self.target_size = target_size

        # Calculate initial channels and spatial size for ConvTranspose
        initial_chans = base_filters * (2**(num_layers - 1))
        # Calculate the spatial size before AdaptiveAvgPool in the encoder
        initial_size = target_size // (2**num_layers)
        if initial_size < 1: initial_size = 1 # Ensure size is at least 1

        self.initial_fc = nn.Linear(feat_dim, initial_chans * initial_size * initial_size)
        self.initial_reshape_size = (initial_chans, initial_size, initial_size)

        layers = []
        current_chans = initial_chans
        for i in range(num_layers - 1, -1, -1): # Iterate backwards
            out_chans = base_filters * (2**(i-1)) if i > 0 else target_chans
            layers.append(nn.ConvTranspose2d(current_chans, out_chans, kernel_size=4, stride=2, padding=1))
            # Add Norm and Activation, except for the last layer
            if i > 0:
                    layers.append(nn.BatchNorm2d(out_chans))
                    layers.append(activation())
            current_chans = out_chans

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # Project and reshape feature vector
        x = self.initial_fc(x)
        x = x.view(-1, *self.initial_reshape_size) # [B, C_init, H_init, W_init]
        # Decode using ConvTranspose layers
        x = self.decoder(x)
        # Ensure output size matches target (might need final interpolation/padding if strides/kernels don't align perfectly)
        # Example using interpolation:
        # x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        return x


# --- Main RNN Sequence Model ---

class NoiseSequenceRNN(nn.Module):
    """
    RNN (GRU) based model to predict a sequence of evolving noise tensors autoregressively,
    conditioned on a text prompt embedding.
    """
    def __init__(self,
                    # Text Embed Config
                    text_embed_dim: int,
                    # Noise I/O Config
                    noise_img_size: int = 128,
                    noise_in_chans: int = 4,
                    # CNN Encoder/Decoder Config
                    cnn_base_filters: int = 32,
                    cnn_num_layers: int = 4, # Adjust based on noise_img_size
                    cnn_feat_dim: int = 512, # Output dim of CNN encoder
                    # GRU Config
                    gru_hidden_size: int = 1024,
                    gru_num_layers: int = 2,
                    gru_dropout: float = 0.1, # Dropout between GRU layers
                    # Output Config
                    predict_variance: bool = True
                ):
        """
        Initializes the NoiseSequenceRNN model.

        Args:
            text_embed_dim: Dimension of pre-computed text embedding.
            noise_img_size: Height/Width of noise tensors.
            noise_in_chans: Input channels of noise.
            cnn_base_filters: Base number of filters for CNN encoder/decoder.
            cnn_num_layers: Number of down/up sampling layers in CNNs.
            cnn_feat_dim: Feature dimension after CNN encoder.
            gru_hidden_size: Hidden dimension of the GRU layers.
            gru_num_layers: Number of GRU layers.
            gru_dropout: Dropout probability between GRU layers.
            predict_variance: If True, predict mean and log_variance.
        """
        super().__init__()
        self.predict_variance = predict_variance
        self.noise_img_size = noise_img_size
        self.noise_in_chans = noise_in_chans
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        # --- Components ---
        # CNN Encoder for noise
        self.noise_encoder = SimpleCNNEncoder(
            in_chans=noise_in_chans,
            base_filters=cnn_base_filters,
            num_layers=cnn_num_layers,
            feat_dim=cnn_feat_dim
        )

        # Linear layer to project text embedding (optional, if needed to match dims)
        # GRU input will be concatenation of noise_feat and text_embed
        gru_input_dim = cnn_feat_dim + text_embed_dim
        # self.text_projection = nn.Linear(text_embed_dim, text_embed_dim) # Example if projection needed

        # GRU Core
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True, # Expect input shape [Batch, SeqLen, FeatureDim]
            dropout=gru_dropout if gru_num_layers > 1 else 0.0 # Add dropout only if multiple layers
        )

        # Output Head (CNN Decoder)
        final_out_channels = noise_in_chans * 2 if predict_variance else noise_in_chans
        self.output_decoder = SimpleCNNDecoder(
            feat_dim=gru_hidden_size, # Input is GRU hidden state
            target_chans=final_out_channels,
            target_size=noise_img_size,
            base_filters=cnn_base_filters, # Use same base filters as encoder
            num_layers=cnn_num_layers     # Use same num layers as encoder
        )

        print(f"Initialized NoiseSequenceRNN:")
        print(f"  - CNN Feature Dim: {cnn_feat_dim}")
        print(f"  - GRU Input Dim: {gru_input_dim}")
        print(f"  - GRU Hidden Size: {gru_hidden_size}, Layers: {gru_num_layers}")
        print(f"  - Predict Variance: {predict_variance}")


    def forward(self, src_noise_sequence, text_embed, initial_hidden_state=None):
        """
        Forward pass for training (using teacher forcing).

        Args:
            src_noise_sequence (torch.Tensor): Sequence of input noise tensors
                                                [B, SeqLen_in, C, H, W].
                                                Example: [x_T, x'_1, ..., x'_{n-1}]
            text_embed (torch.Tensor): Text embedding [B, D_txt] (assuming pooled/single vector).
            initial_hidden_state (torch.Tensor, optional): Initial hidden state for GRU.

        Returns:
            If predict_variance:
                tuple(torch.Tensor, torch.Tensor): Predicted mean and log_variance sequences
                                                    [B, SeqLen_out, C, H, W]. SeqLen_out = SeqLen_in.
            Else:
                torch.Tensor: Predicted mean sequence [B, SeqLen_out, C, H, W].
        """
        B, T_in, C, H, W = src_noise_sequence.shape
        device = src_noise_sequence.device

        # --- 1. Encode Noise Sequence ---
        # Reshape for CNN: [B * T_in, C, H, W]
        noise_flat_batch = src_noise_sequence.view(B * T_in, C, H, W)
        # Encode each noise tensor: [B * T_in, D_feat]
        noise_features_flat = self.noise_encoder(noise_flat_batch)
        # Reshape back to sequence: [B, T_in, D_feat]
        noise_features_seq = noise_features_flat.view(B, T_in, -1)

        # --- 2. Prepare GRU Input ---
        # Assume text_embed is [B, D_txt]. Expand for sequence length.
        if text_embed.dim() == 2:
                text_embed_expanded = text_embed.unsqueeze(1).expand(-1, T_in, -1) # [B, T_in, D_txt]
        elif text_embed.dim() == 3 and text_embed.shape[1] == 1: # If [B, 1, D_txt]
                text_embed_expanded = text_embed.expand(-1, T_in, -1)
        else:
                # If text_embed is already sequential [B, T_in, D_txt]? Less likely.
                # Handle error or adapt based on actual text_embed shape.
                raise ValueError(f"Unsupported text_embed shape: {text_embed.shape}. Expected [B, D_txt] or [B, 1, D_txt]")

        # Concatenate noise features and text features
        gru_input_seq = torch.cat([noise_features_seq, text_embed_expanded], dim=-1) # [B, T_in, D_feat + D_txt]

        # --- 3. Pass through GRU ---
        # gru_output shape: [B, T_in, D_hidden]
        # final_hidden_state shape: [NumLayers, B, D_hidden]
        gru_output, final_hidden_state = self.gru(gru_input_seq, initial_hidden_state)

        # --- 4. Decode GRU Output Sequence ---
        # Reshape GRU output for CNN decoder: [B * T_in, D_hidden]
        gru_output_flat = gru_output.reshape(B * T_in, self.gru_hidden_size)
        # Decode each step's hidden state: [B * T_in, C_out, H, W]
        decoded_params_flat = self.output_decoder(gru_output_flat)
        # Reshape back to sequence: [B, T_in, C_out, H, W]
        output_sequence = decoded_params_flat.view(B, T_in, -1, H, W)

        # --- 5. Separate Mean and Log Variance ---
        if self.predict_variance:
            mean_seq, log_var_seq = torch.chunk(output_sequence, 2, dim=2) # Split channel dim
            log_var_seq = torch.clamp(log_var_seq, min=-10, max=10)
            return mean_seq, log_var_seq, final_hidden_state # Return hidden state for autoregressive generation
        else:
            return output_sequence, final_hidden_state

    def generate_sequence(self, initial_noise, text_embed, num_steps):
        """
        Generates a noise sequence autoregressively during inference.

        Args:
            initial_noise (torch.Tensor): The starting noise x_T [B, C, H, W].
            text_embed (torch.Tensor): Text embedding [B, D_txt].
            num_steps (int): The number of steps (N) to generate.

        Returns:
            torch.Tensor: The generated sequence of noise tensors [B, N, C, H, W].
        """
        self.eval() # Ensure model is in eval mode
        B, C, H, W = initial_noise.shape
        device = initial_noise.device
        generated_sequence = []
        current_noise_input = initial_noise # Start with x_T
        hidden_state = None # Initial hidden state (defaults to zeros in GRU)

        with torch.no_grad():
            for _ in range(num_steps):
                # Prepare input for current step (unsqueeze to add sequence dim of 1)
                current_noise_seq = current_noise_input.unsqueeze(1) # [B, 1, C, H, W]

                # Forward pass for one step
                if self.predict_variance:
                    mean_pred, log_var_pred, hidden_state = self.forward(current_noise_seq, text_embed, hidden_state)
                    # Sample from predicted distribution
                    std_dev = torch.exp(0.5 * log_var_pred) # [B, 1, C, H, W]
                    sampled_noise = mean_pred + torch.randn_like(mean_pred) * std_dev
                else:
                    mean_pred, hidden_state = self.forward(current_noise_seq, text_embed, hidden_state)
                    sampled_noise = mean_pred # Use mean directly if not predicting variance

                # Get the predicted noise for this step (remove sequence dim)
                next_noise = sampled_noise.squeeze(1) # [B, C, H, W]
                generated_sequence.append(next_noise)
                # Update input for next step
                current_noise_input = next_noise

        # Stack the generated sequence
        return torch.stack(generated_sequence, dim=1) # [B, N, C, H, W]


# Example Usage (if run directly)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Config (Example)
    text_dim = 768
    noise_size = 128
    noise_chans = 4
    cnn_layers = 4 # Should be appropriate for 128 -> small spatial dim
    cnn_feat = 512
    gru_hidden = 1024
    gru_layers = 2
    seq_len = 5 # Example sequence length (initial + 4 steps)
    batch_size = 2

    # Instantiate model
    model = NoiseSequenceRNN(
        text_embed_dim=text_dim,
        noise_img_size=noise_size,
        noise_in_chans=noise_chans,
        cnn_num_layers=cnn_layers,
        cnn_feat_dim=cnn_feat,
        gru_hidden_size=gru_hidden,
        gru_num_layers=gru_layers,
        predict_variance=True
    ).to(device)
    model.eval()

    # Create dummy inputs for training forward pass
    dummy_src_noise_seq = torch.randn(batch_size, seq_len, noise_chans, noise_size, noise_size).to(device)
    dummy_text_embed_train = torch.randn(batch_size, text_dim).to(device)

    print("\nTesting NoiseSequenceRNN (Training Forward)...")
    print("Input Noise Seq Shape:", dummy_src_noise_seq.shape)
    print("Input Text Embed Shape:", dummy_text_embed_train.shape)

    with torch.no_grad():
        mean_pred_seq, log_var_pred_seq, _ = model(dummy_src_noise_seq, dummy_text_embed_train)

    print("Output Mean Seq Shape:", mean_pred_seq.shape)
    print("Output LogVar Seq Shape:", log_var_pred_seq.shape)

    # Test generation
    print("\nTesting NoiseSequenceRNN (Generation)...")
    dummy_initial_noise = torch.randn(batch_size, noise_chans, noise_size, noise_size).to(device)
    dummy_text_embed_gen = torch.randn(batch_size, text_dim).to(device)
    num_gen_steps = 4

    with torch.no_grad():
        generated_seq = model.generate_sequence(dummy_initial_noise, dummy_text_embed_gen, num_gen_steps)

    print("Generated Sequence Shape:", generated_seq.shape) # Expected [B, num_gen_steps, C, H, W]

