"""  Attention and normalization modules  """
from core.modules.util_class import Elementwise
from core.modules.gate import context_gate_factory, ContextGate
from core.modules.global_attention import GlobalAttention
from core.modules.conv_multi_step_attention import ConvMultiStepAttention
from core.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, \
    CopyGeneratorLossCompute
from core.modules.multi_headed_attn import MultiHeadedAttention
from core.modules.embeddings import Embeddings, PositionalEncoding, \
    VecEmbedding
from core.modules.weight_norm import WeightNormConv2d
from core.modules.average_attn import AverageAttention
from core.modules.dual_attention import DualAttention

__all__ = ["Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "ConvMultiStepAttention", "CopyGenerator",
           "CopyGeneratorLoss", "CopyGeneratorLossCompute",
           "MultiHeadedAttention", "Embeddings", "PositionalEncoding",
           "WeightNormConv2d", "AverageAttention", "VecEmbedding",
           "DualAttention"]
