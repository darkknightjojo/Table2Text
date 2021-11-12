import torch
from torch import nn
from ..modules import GlobalAttention


class EmptyDecoder(nn.Module):

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            hidden_size=opt.rnn_size,
            num_layers=opt.dec_layers,
            bidirectional_encoder=opt.brnn,
            attn_type=opt.global_attention,
            attn_func=opt.global_attention_function,
            coverage_attn=opt.coverage_attn,
            copy_attn=opt.copy_attn,
            dropout=opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            embeddings=embeddings,
            reuse_copy_attn=opt.reuse_copy_attn,
            copy_attn_type=opt.copy_attn_type)

    def __init__(self, hidden_size, num_layers,
                 bidirectional_encoder, embeddings=None,
                 attn_type="general", attn_func="softmax",
                 coverage_attn=False, copy_attn=False,
                 reuse_copy_attn=False, copy_attn_type="general",
                 dropout=0.0, attentional=True):

        super(EmptyDecoder, self).__init__()
        self.attentional = attentional
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.bidirectional_encoder = bidirectional_encoder
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

    def forward(self, dec_in, memory_bank, memory_lengths, rnn, **dec_kwargs):
        weights = dec_kwargs.pop('weights', None)
        assert weights is not None

        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()

        dec_outs = []
        attns = {}

        if self.attn is not None:
            attns["std"] = []
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(dec_in)
        assert emb.dim() == 3  # len * batch * embedding_dim

        dec_states = self._unlink_states(self.state['hidden'])

        # Input feed concatenates hidden state with input at every time step.
        for idx, emb_t in enumerate(emb.split(1)):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            new_states = list()
            rnn_output, dec_states = rnn(decoder_input, dec_states)

        if self.attentional:
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths
            )
        else:
            decoder_output = rnn_output;

        decoder_output = self.dropout(decoder_output)
        input_feed = decoder_output

        dec_outs += [decoder_output]

        if self.copy_attn is not None:
            _, copy_attn = self.copy_attn(decoder_output, memory_bank.transpose(0, 1))
            attns["copy"] += [copy_attn]
        elif self._reuse_copy_attn:
            attns["copy"] = attns["std"]

        return self._link_states(dec_states), dec_outs, attns

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

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""

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
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final),)

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None
