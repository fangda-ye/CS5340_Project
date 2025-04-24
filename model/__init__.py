# model/__init__.py

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

if SEQ_RNN_AVAILABLE:
    __all__.extend([
        'NoiseSequenceRNN_v3',
        'ResBlock', 'ResNetCNNEncoder',
        'ResNetCNNDecoderFiLM', 'FiLMLayer'
    ])

