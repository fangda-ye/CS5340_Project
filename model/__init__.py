# model/__init__.py

# --- Original NPNet Components ---
try:
    from .Attention import Attention
    from .NoiseTransformer import NoiseTransformer
    from .SVDNoiseUnet import SVDNoiseUnet, SVDNoiseUnet_Concise
    from .npnet import NPNet, CrossAttention
    ORIGINAL_NPNet_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import original NPNet components: {e}")
    # Define dummy classes if needed
    class Attention: pass
    class NoiseTransformer: pass
    class SVDNoiseUnet: pass
    class SVDNoiseUnet_Concise: pass
    class NPNet: pass
    class CrossAttention: pass
    ORIGINAL_NPNet_AVAILABLE = False

# --- Transformer Sequence Model Components ---
try:
    # Assuming seq_model.py contains these
    from .seq_model import NoiseSequenceTransformer, PatchEmbedding, PositionalEncoding
    SEQ_TRANSFORMER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Transformer sequence model components: {e}")
    class NoiseSequenceTransformer: pass
    class PatchEmbedding: pass
    class PositionalEncoding: pass
    SEQ_TRANSFORMER_AVAILABLE = False

# --- RNN Sequence Model Components ---
try:
    # Assuming rnn_seq_model_v3.py contains both RNN versions
    from .rnn_seq_model_v3 import NoiseSequenceRNN_v3
    # Also import helpers if they are defined there and needed externally
    from .rnn_seq_model_v3 import ResBlock, ResNetCNNEncoder, ResNetCNNDecoderFiLM, FiLMLayer
    SEQ_RNN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RNN sequence model components: {e}")
    class NoiseSequenceRNN_v3: pass
    class ResBlock: pass
    class ResNetCNNEncoder: pass
    class ResNetCNNDecoderFiLM: pass
    class FiLMLayer: pass
    SEQ_RNN_AVAILABLE = False


# Define what gets imported with "from model import *"
__all__ = []

if ORIGINAL_NPNet_AVAILABLE:
    __all__.extend([
        'Attention', 'NoiseTransformer', 'SVDNoiseUnet',
        'SVDNoiseUnet_Concise', 'CrossAttention', 'NPNet'
    ])

if SEQ_TRANSFORMER_AVAILABLE:
    __all__.extend([
        'NoiseSequenceTransformer', 'PatchEmbedding', 'PositionalEncoding'
    ])

if SEQ_RNN_AVAILABLE:
    __all__.extend([
        'NoiseSequenceRNN_v3',
        'ResBlock', 'ResNetCNNEncoder',
        'ResNetCNNDecoderFiLM', 'FiLMLayer'
    ])

# You can print the final list for verification
# print(f"model/__init__.py: Exporting {__all__}")

