""" module NMT Model base class definition """
import torch.nn as nn


class T2TModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (module.encoder.EncoderBase): an encoder object
      decoder (module.decoder.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(T2TModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, **kwargs):
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
        
        # separate additionnal args for encoder/decoder
        enc_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('enc')}
        dec_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('dec')}

        # enc_state = feature_size * batch_size * hidden_size
        # memory_bank = src_length * batch_size * hidden_size
        enc_state, memory_bank, lengths = self.encoder(src, lengths, **enc_kwargs)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state, **dec_kwargs)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align,
                                      **dec_kwargs)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
