""" module NMT Model base class definition """
import torch.nn as nn
from core.rnn.LSTM import LSTM
from core.decoder.empty_decoder import EmptyDecoder


class SwitchModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (module.encoder.EncoderBase): an encoder object
      decoder (module.decoder.DecoderBase): a decoder object
    """

    def __init__(self, opt, src_embeddings, tgt_embeddings, dim=None):
        super(SwitchModel, self).__init__()
        self.rnn1 = LSTM.from_opt(opt, src_embeddings)
        self.rnn2 = LSTM.from_opt(opt, tgt_embeddings)
        self.encoder = self.rnn1
        self.decoder = EmptyDecoder.from_opt(opt, src_embeddings, tgt_embeddings)

    def forward(self, src, tgt, lengths=None, bptt=False, with_align=False, reverse=False, **kwargs):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            dec_kwargs: 权重文件或者表格编码
            reverse: 是否交换RNN
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        target = tgt[:-1]  # exclude last target from inputs

        rnn_encoder = self.rnn1
        rnn_decoder = self.rnn2
        if self.training:
            # 交换RNN
            if reverse:
                rnn_encoder = self.rnn2
                rnn_decoder = self.rnn1

        dec_kwargs = {"embeddings": {key[6:]: value for key, value in kwargs.items() if key.startswith('table')}}
        # 将table_embeddings转为embeddings

        enc_state, memory_bank, lengths = encoder_forward(src, lengths=None, rnn=rnn_encoder)

        if bptt is False:
            if reverse:
                self.decoder.init_state(memory_bank=memory_bank, encoder_final=enc_state)
            else:
                self.decoder.init_state(memory_bank=memory_bank, encoder_final=enc_state, **dec_kwargs)

        dec_states, dec_out, attns = self.decoder(target, memory_bank, memory_lengths=lengths,
                                                  rnn=rnn_decoder, reverse=reverse, **dec_kwargs)

        return dec_out, attns

    def update_dropout(self, dropout):
        self.rnn1.update_dropout(dropout)
        self.rnn2.update_dropout(dropout)


def encoder_forward(src, lengths, rnn, **enc_kwargs):
    return rnn(src, lengths, **enc_kwargs)


def decoder_forward(decoder, target, memory_bank, lengths, rnn, reverse, **dec_kwargs):
    return decoder(target, memory_bank, memory_lengths=lengths, rnn=rnn, reverse=reverse, **dec_kwargs)
