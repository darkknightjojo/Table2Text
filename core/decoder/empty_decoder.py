import sys

import torch
from torch import nn
from ..modules import GlobalAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmptyDecoder(nn.Module):

    @classmethod
    def from_opt(cls, opt, src_embeddings, embeddings):
        """Alternate constructor."""
        return cls(
            hidden_size=opt.rnn_size,
            bidirectional_encoder=opt.brnn,
            attn_type=opt.global_attention,
            attn_func=opt.global_attention_function,
            coverage_attn=opt.coverage_attn,
            copy_attn=opt.copy_attn,
            dropout=opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            embeddings=embeddings,
            reuse_copy_attn=opt.reuse_copy_attn,
            copy_attn_type=opt.copy_attn_type,
            src_embeddings=src_embeddings,
            num_layers=opt.layers)

    def __init__(self, hidden_size, embeddings=None, src_embeddings=None, bidirectional_encoder=None, num_layers=2,
                 attn_type="general",
                 attn_func="softmax",
                 coverage_attn=False, copy_attn=False, reuse_copy_attn=False, copy_attn_type="general",
                 dropout=0.0, attentional=True):

        super(EmptyDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.src_embeddings = src_embeddings
        self.dropout = nn.Dropout(dropout)

        self.attentional = attentional

        # Decoder state
        self.state = {}
        # Set up the standard attention.
        self._coverage = coverage_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            self.attn = None
        else:
            self.attn = GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type, attn_func=attn_func
            )

        if copy_attn and not reuse_copy_attn:
            if copy_attn_type == "none" or copy_attn_type is None:
                raise ValueError(
                    "Cannot use copy_attn with copy_attn_type none")
            self.copy_attn = GlobalAttention(
                hidden_size, attn_type=copy_attn_type, attn_func=attn_func
            )
        else:
            self.copy_attn = None

        self._reuse_copy_attn = reuse_copy_attn and copy_attn
        if self._reuse_copy_attn and not self.attentional:
            raise ValueError("Cannot reuse copy attention with no attention.")

        # 线性层，将tabbie的embeddings映射为768
        self.table_embedding_relu = nn.ReLU()
        self.table_embedding_row_fw_input = nn.Linear(3 * 768, 2*768)
        self.table_embedding_row_fw_output = nn.Linear(2*768, 768)
        self.table_embedding_col_fw_input = nn.Linear(3 * 768, 2 * 768)
        self.table_embedding_col_fw_output = nn.Linear(2 * 768, 768)

    def init_state(self, src=None, memory_bank=None, encoder_final=None, **kwargs):
        """Initialize decoder state with last state of the encoder."""
        # assert encoder_final is not None

        if kwargs is not None:
            if kwargs['embeddings'] is not None:
                # 使用tabbie的输出作为rnn初始化的张量
                # batch_size * 768
                table_embeddings = kwargs.pop('embeddings', None)
                new_table_embeddings = []
                if table_embeddings is not None:
                    if isinstance(table_embeddings, dict):
                        table_embeddings = table_embeddings.get('embeddings', None)
                    # 使用线性层将table_embeddings 映射到 768
                    if table_embeddings is not None:
                        for table_embedding in table_embeddings:
                            e = self.map_embedding(table_embedding)
                            new_table_embeddings.append(e)
                encoder_final = new_table_embeddings

        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final)
        elif isinstance(encoder_final, list):  # tabbie
            batch_size = len(encoder_final)
            row_embeddings = []
            col_embeddings = []
            for item in encoder_final:
                row_embeddings.append(item[0])
                col_embeddings.append(item[1])
            if len(row_embeddings) == 0 or len(col_embeddings) == 0:
                print("爷不完了！！")
                sys.exit(-1)
            hidden0 = torch.stack(row_embeddings).reshape((batch_size, 768)).repeat((self.num_layers, 1, 1))
            hidden1 = torch.stack(col_embeddings).reshape((batch_size, 768)).repeat((self.num_layers, 1, 1))
            self.state["hidden"] = tuple([hidden0, hidden1])
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final),)

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        # decoder第一个输入向量的初始化
        self.state["input_feed"] = self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def map_embedding(self, embeddings):

        assert isinstance(embeddings, tuple)

        row_embeddings_1 = self.table_embedding_row_fw_input(embeddings[0].reshape(1, 3 * 768))
        row_embeddings_2 = self.table_embedding_relu(row_embeddings_1)
        row_embeddings_final = self.table_embedding_row_fw_output(row_embeddings_2).squeeze()

        col_embeddings_1 = self.table_embedding_col_fw_input(embeddings[1].reshape(1, 3 * 768))
        col_embeddings_2 = self.table_embedding_relu(col_embeddings_1)
        col_embeddings_final = self.table_embedding_col_fw_output(col_embeddings_2).squeeze()

        return row_embeddings_final, col_embeddings_final

    def forward(self, target, memory_bank, memory_lengths, rnn, reverse=False, **dec_kwargs):

        # input_feed.shape: batch_size * hidden_size
        input_feed = self.state["input_feed"]

        dec_outs = []
        attns = {}

        if self.attn is not None:
            attns["std"] = []
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        if self.src_embeddings is not None and reverse:
            tgt_emb = self.src_embeddings(target)
        else:
            tgt_emb = self.embeddings(target)

        assert tgt_emb.dim() == 3  # len * batch * embedding_dim

        dec_states = self.state['hidden']

        # rnn_output_list, dec_states_list = rnn(target, hidden=dec_states, is_decoder=True, input_feed=input_feed)
        # Input feed concatenates hidden state with input at every time step.
        for idx, emb_t in enumerate(tgt_emb.split(1)):
            decoder_input = torch.cat([emb_t, input_feed], 2)
            rnn_output, dec_states = rnn(input=decoder_input, hidden=dec_states, is_decoder=True)

            if self.attentional:
                decoder_output, p_attn = self.attn(
                    rnn_output.transpose(0, 1),
                    memory_bank.transpose(0, 1),
                    memory_lengths=memory_lengths
                )
                attns["std"].append(p_attn)
            else:
                decoder_output = rnn_output

            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output
            dec_outs += [decoder_output]

            if self.copy_attn is not None:
                _, copy_attn = self.copy_attn(decoder_output, memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._reuse_copy_attn:
                attns["copy"] = attns["std"]

        # Update the state with the result.
        if not isinstance(dec_states, tuple):
            dec_states = (dec_states,)
        self.state["hidden"] = dec_states
        self.state["input_feed"] = dec_outs[-1]
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].squeeze(0)
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
                    attns[k] = attns[k].squeeze(dim=1)
        dec_outs = dec_outs.squeeze(dim=1)

        return dec_states, dec_outs, attns

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def map_state(self, fn, select_indices=None):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(self.state["coverage"], 1)
