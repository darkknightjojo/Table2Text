"""  Attention and normalization modules  """
from module.modules.util_class import Elementwise
from module.modules.gate import context_gate_factory, ContextGate
from module.modules.global_attention import GlobalAttention
from module.modules.conv_multi_step_attention import ConvMultiStepAttention
from module.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, \
    CopyGeneratorLossCompute
from module.modules.multi_headed_attn import MultiHeadedAttention
from module.modules.embeddings import Embeddings, PositionalEncoding, \
    VecEmbedding
from module.modules.weight_norm import WeightNormConv2d
from module.modules.average_attn import AverageAttention
from module.modules.dual_attention import DualAttention

__all__ = ["Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "ConvMultiStepAttention", "CopyGenerator",
           "CopyGeneratorLoss", "CopyGeneratorLossCompute",
           "MultiHeadedAttention", "Embeddings", "PositionalEncoding",
           "WeightNormConv2d", "AverageAttention", "VecEmbedding",
           "DualAttention"]
