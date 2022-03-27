""" module NMT Model base class definition """
import torch
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (module.encoder.EncoderBase): an encoder object
      decoder (module.decoder.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, row_linear=None, col_linear=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # 线性层，将tabbie的embeddings映射为768
        self.table_embedding_row_linear = row_linear
        self.table_embedding_col_linear = col_linear

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
        # 将dec_weights转为weights
        dec_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('dec')}
        # 将table_embeddings转为embeddings
        table_embeddings = {key[6:]: value for key, value in kwargs.items() if key.startswith('table')}

        # 使用线性层将table_embeddings 映射到 768
        new_table_embeddings = []
        for table_embedding in table_embeddings.pop('embeddings'):
            e = self.map_embedding(table_embedding)
            new_table_embeddings.append(e)
        dec_kwargs["embeddings"] = new_table_embeddings

        # enc_state = layer_num * batch_size * hidden_size 保存了每一个时间步的hidden和cell
        # memory_bank = src_length * batch_size * hidden_size 最后一层的输出
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

    def map_embedding(self, embeddings):

        assert isinstance(embeddings, tuple)

        row_embeddings = self.table_embedding_row_linear(embeddings[0].reshape(1, 3 * 768)).squeeze()
        col_embeddings = self.table_embedding_col_linear(embeddings[1].reshape(1, 3 * 768)).squeeze()

        return row_embeddings, col_embeddings