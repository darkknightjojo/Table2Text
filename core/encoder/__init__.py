from core.encoder.encoder import EncoderBase
from core.encoder.rnn_encoder import RNNEncoder
from core.encoder.structure_aware import StructureAwareEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "LSTM": RNNEncoder}

__all__ = ["EncoderBase", "RNNEncoder", "str2enc"]