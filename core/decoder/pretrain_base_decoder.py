import torch
import torch.nn as nn

from core.models.stacked_rnn import StackedLSTM, StackedGRU
from core.modules import context_gate_factory, GlobalAttention
from core.utils.rnn_factory import rnn_factory

from core.utils.misc import aeq
from core.decoder.decoder import RNNDecoderBase


class PretrainBaseRNNDecoder(RNNDecoderBase):
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, nb_branches,
                 attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, branch_dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, copy_attn_type="general"):
        super(RNNDecoderBase, self).__init__(
            attentional=attn_type != "none" and attn_type is not None)

        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.branch_dropout = branch_dropout

        # Decoder state
        self.state = {}

        # Build the RNNs.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

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


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            rnn_type=opt.rnn_type,
            bidirectional_encoder=opt.brnn,
            num_layers=opt.dec_layers,
            hidden_size=opt.dec_rnn_size,
            nb_branches=opt.nb_branches,
            attn_type=opt.global_attention,
            attn_func=opt.global_attention_function,
            coverage_attn=opt.coverage_attn,
            context_gate=opt.context_gate,
            copy_attn=opt.copy_attn,
            dropout=opt.dropout[0] if isinstance(opt.dropout, list) else opt.dropout,
            branch_dropout=opt.branch_dropout,
            embeddings=embeddings,
            reuse_copy_attn=opt.reuse_copy_attn,
            copy_attn_type=opt.copy_attn_type)

    def init_state(self, src, memory_bank, encoder_final, **kwargs):
        """
        This is a bit tricky: states is either a tensor or tuple of tensors.
        Furthermore, OpenNMT needs them for stuff, especially during translation.
        We therefore cannot return simply a list of tensors/tuples because all
        their funcs won't work.

        Therefore we cat all hidden_states (and cell_states) on the last dim (features dim)
        We will be able to _unlink_states later when needed inside the _run_forward_pass
        """
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

        # encoder_final的形状应当为layers * batch_size * hidden_num
        super().init_state(src, memory_bank, encoder_final)
        repeats = [1] * len(self.state['hidden'][0].shape)
        self.state['hidden'] = tuple(h.repeat(*repeats) for h in self.state['hidden'])

    def map_embedding(self, embeddings):

        assert isinstance(embeddings, tuple)

        row_embeddings_1 = self.table_embedding_row_fw_input(embeddings[0].reshape(1, 3 * 768))
        row_embeddings_2 = self.table_embedding_relu(row_embeddings_1)
        row_embeddings_final = self.table_embedding_row_fw_output(row_embeddings_2).squeeze()

        col_embeddings_1 = self.table_embedding_col_fw_input(embeddings[1].reshape(1, 3 * 768))
        col_embeddings_2 = self.table_embedding_relu(col_embeddings_1)
        col_embeddings_final = self.table_embedding_col_fw_output(col_embeddings_2).squeeze()

        return row_embeddings_final, col_embeddings_final

    def _link_states(self, dec_states):
        if isinstance(dec_states[0], tuple):
            left = torch.cat([left for left, right in dec_states], dim=-1)
            right = torch.cat([right for left, right in dec_states], dim=-1)
            return left, right
        return torch.cat(dec_states, dim=-1)

    def _unlink_states(self, dec_states):
        if isinstance(dec_states, tuple):
            dec_states = [state.chunk(len(self.rnns), dim=-1)
                          for state in dec_states]
            return list(zip(*dec_states))
        else:
            return dec_states.chunk(len(self.rnns), dim=-1)

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None, **kwargs):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """

        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]

        # Input feed concatenates hidden state with
        # input at every time step.
        for idx, emb_t in enumerate(emb.split(1)):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)

            rnn_output, dec_state = self.rnn(decoder_input, dec_state)

            if self.attentional:
                decoder_output, p_attn = self.attn(
                    rnn_output,
                    memory_bank.transpose(0, 1),
                    memory_lengths=memory_lengths)
                attns["std"].append(p_attn)
            else:
                decoder_output = rnn_output
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]

            # Update the coverage attention.
            if self._coverage:
                coverage = p_attn if coverage is None else p_attn + coverage
                attns["coverage"] += [coverage]

            if self.copy_attn is not None:
                _, copy_attn = self.copy_attn(
                    decoder_output, memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._reuse_copy_attn:
                attns["copy"] = attns["std"]

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert rnn_type != "SRU", "SRU doesn't support input feed! " \
                                  "Please set -input_feed 0!"
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)
