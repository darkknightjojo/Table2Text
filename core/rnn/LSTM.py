import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
from torch.nn.functional import pad

class LSTM(nn.Module):

    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, dropout=0.0,
                 embeddings=None, use_bridge=False):
        super(LSTM, self).__init__()
        assert embeddings is not None

        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.embeddings = embeddings

        self.rnn = nn.LSTM(input_size=embeddings.embedding_size + hidden_size,
                           hidden_size=self.hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)

        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type, hidden_size, num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge)

    def forward(self, input, lengths=None, hidden=None, is_decoder=False):

        # s_len, batch, emb_dim = emb.size()
        if not is_decoder:
            emb = self.embeddings(input)
            packed_emb = emb
            # 填充为 len* batch_size * hidden_size+embedding
            packed_emb = pad(packed_emb, (self.hidden_size * self.num_directions, 0))
            # 压缩 去除无效0
            if lengths is not None:
                lengths_list = lengths.view(-1).tolist()
                packed_emb = pack(packed_emb, lengths_list)

            # TODO memory_bank 是什么？
            memory_bank, encoder_final = self.rnn(packed_emb)
            # 填充0 保证长度一致
            if lengths is not None:
                memory_bank = unpack(memory_bank)[0]
            #  switch rnn 不需要使用bridge
            if self.use_bridge:
                encoder_final = self._bridge(encoder_final)
            return encoder_final, memory_bank, lengths

        else:
            # decoder直接传进来embedding
            output, state = self.rnn(input, hidden)
            return output, state


    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):
        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs