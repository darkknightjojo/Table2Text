"""core defining decoder."""
from core.decoder.decoder import DecoderBase, InputFeedRNNDecoder, StdRNNDecoder
from core.decoder.multi_branch_decoder import MultiBranchRNNDecoder
from core.decoder.structure_aware import StructureAwareDecoder
from core.decoder.pretrain_base_decoder import PretrainBaseRNNDecoder
from core.decoder.decoder_with_lm import MultiBranchWithLMDecoder

str2dec = {"rnn": StdRNNDecoder, "brnn": StdRNNDecoder, "LSTM": StdRNNDecoder,
           "pbrnn": PretrainBaseRNNDecoder, "mbrnn": MultiBranchRNNDecoder, "mblm": MultiBranchWithLMDecoder}

__all__ = ["DecoderBase", "StdRNNDecoder", "InputFeedRNNDecoder", "str2dec"]
