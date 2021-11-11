""" module NMT Model base class definition """
import torch.nn as nn
from module.encoder import str2enc
from module.decoder import str2dec


class SwitchModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (module.encoder.EncoderBase): an encoder object
      decoder (module.decoder.DecoderBase): a decoder object
    """

    def __init__(self, opt, embeddings, dim=None):
        super(SwitchModel, self).__init__()
        self.rnn1 = str2enc[opt.rnn_type].from_opt(opt, embeddings)
        self.rnn2 = str2enc[opt.rnn_type].from_opt(opt, embeddings)
        # self.encoder = encoder
        # self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, reverse=False, **kwargs):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
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
        dec_in = tgt[:-1]  # exclude last target from inputs

        # 交换RNN
        if reverse:
            rnn_encoder = self.rnn2
            rnn_decoder = self.rnn1
            # TODO 不确定这里的参数是否应该进行交换
            # separate additionnal args for encoder/decoder
            enc_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('dec')}
            dec_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('enc')}
        else:
            rnn_encoder = self.rnn1
            rnn_decoder = self.rnn2
            # separate additionnal args for encoder/decoder
            enc_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('enc')}
            dec_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('dec')}

        # # separate additionnal args for encoder/decoder
        # enc_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('enc')}
        # dec_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('dec')}

        enc_state, memory_bank, lengths = encoder_forward(src, lengths, rnn_encoder, **enc_kwargs)

        decoder_init(rnn_decoder, src, memory_bank, enc_state)

        dec_out, attns = decoder_forward(rnn_decoder, dec_in, memory_bank, lengths=lengths,with_align=with_align, **dec_kwargs)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.rnn1.update_dropout(dropout)
        self.rnn2.update_dropout(dropout)


def encoder_forward(src, lengths, rnn, **enc_kwargs):
    return rnn(src, lengths, **enc_kwargs)


def decoder_init(rnn, src, memory_bank, enc_state):
    rnn.init_state(src, memory_bank, enc_state)


def decoder_forward(rnn, dec_in, memory_bank, lengths, with_align, **dec_kwargs):
    return rnn(dec_in, memory_bank, memory_lengths=lengths, with_align=with_align, **dec_kwargs)

