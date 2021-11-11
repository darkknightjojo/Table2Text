"""Module defining decoder."""
from module.decoder.decoder import DecoderBase, InputFeedRNNDecoder, StdRNNDecoder
from module.decoder.multi_branch_decoder import MultiBranchRNNDecoder
from module.decoder.structure_aware import StructureAwareDecoder


str2dec = {"rnn": StdRNNDecoder, "brnn": StdRNNDecoder}

__all__ = ["DecoderBase", "StdRNNDecoder", "InputFeedRNNDecoder", "str2dec"]