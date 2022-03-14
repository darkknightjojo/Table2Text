"""core defining decoder."""
from core.decoder.decoder import DecoderBase, InputFeedRNNDecoder, StdRNNDecoder
from core.decoder.multi_branch_decoder import MultiBranchRNNDecoder
from core.decoder.structure_aware import StructureAwareDecoder
from core.decoder.pretrain_base_decoder import PretrainBaseRNNDecoder

str2dec = {"rnn": StdRNNDecoder, "brnn": StdRNNDecoder, "LSTM": StdRNNDecoder,
           "pbrnn": PretrainBaseRNNDecoder, "mbrnn": MultiBranchRNNDecoder}

__all__ = ["DecoderBase", "StdRNNDecoder", "InputFeedRNNDecoder", "str2dec"]
