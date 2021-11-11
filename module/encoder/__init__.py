from module.encoder.encoder import EncoderBase
from module.encoder.rnn_encoder import RNNEncoder
from module.encoder.structure_aware import StructureAwareEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "sarnn": StructureAwareEncoder}

__all__ = ["EncoderBase", "RNNEncoder", "str2enc", "StructureAwareEncoder"]