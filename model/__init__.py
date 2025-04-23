# # model/__init__.py

# # Import classes from sibling files to make them available when importing 'model'
# from .Attention import Attention, DropPath # Assuming DropPath is also in Attention.py or imported there
# from .NoiseTransformer import NoiseTransformer
# from .SVDNoiseUnet import SVDNoiseUnet, SVDNoiseUnet_Concise
# from .npnet import NPNet, CrossAttention # Import the refactored NPNet and CrossAttention
# from .seq_model import NoiseSequenceTransformer, PatchEmbedding, PositionalEncoding

# # Define what gets imported with "from model import *"
# __all__ = [
#     'Attention',
#     'DropPath', # Export DropPath if used externally
#     'NoiseTransformer',
#     'SVDNoiseUnet',
#     'SVDNoiseUnet_Concise',
#     'CrossAttention', # Export CrossAttention
#     'NPNet'
#     'NoiseSequenceTransformer',
#     'PatchEmbedding',
#     'PositionalEncoding',
# ]

# model/__init__.py

# --- Original NPNet Components ---
try:
    from .Attention import Attention, DropPath
    from .NoiseTransformer import NoiseTransformer
    from .SVDNoiseUnet import SVDNoiseUnet, SVDNoiseUnet_Concise
    from .npnet import NPNet, CrossAttention
    ORIGINAL_NPNet_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import original NPNet components: {e}")
    # Define dummy classes if needed
    class Attention: pass
    class DropPath: pass
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
    # Assuming rnn_seq_model_v2.py contains both RNN versions
    from .rnn_seq_model_v2 import NoiseSequenceRNN, NoiseSequenceRNN_v2
    # Also import helpers if they are defined there and needed externally
    from .rnn_seq_model_v2 import SimpleCNNEncoder, SimpleCNNDecoder, ResBlock, ResNetCNNEncoder, ResNetCNNDecoderFiLM, FiLMLayer
    SEQ_RNN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RNN sequence model components: {e}")
    class NoiseSequenceRNN: pass
    class NoiseSequenceRNN_v2: pass
    class SimpleCNNEncoder: pass
    class SimpleCNNDecoder: pass
    class ResBlock: pass
    class ResNetCNNEncoder: pass
    class ResNetCNNDecoderFiLM: pass
    class FiLMLayer: pass
    SEQ_RNN_AVAILABLE = False


# Define what gets imported with "from model import *"
__all__ = []

if ORIGINAL_NPNet_AVAILABLE:
    __all__.extend([
        'Attention', 'DropPath', 'NoiseTransformer', 'SVDNoiseUnet',
        'SVDNoiseUnet_Concise', 'CrossAttention', 'NPNet'
    ])

if SEQ_TRANSFORMER_AVAILABLE:
    __all__.extend([
        'NoiseSequenceTransformer', 'PatchEmbedding', 'PositionalEncoding'
    ])

if SEQ_RNN_AVAILABLE:
    __all__.extend([
        'NoiseSequenceRNN', 'NoiseSequenceRNN_v2', 'SimpleCNNEncoder',
        'SimpleCNNDecoder', 'ResBlock', 'ResNetCNNEncoder',
        'ResNetCNNDecoderFiLM', 'FiLMLayer'
    ])

# You can print the final list for verification
# print(f"model/__init__.py: Exporting {__all__}")

