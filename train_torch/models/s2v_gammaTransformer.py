import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modified_mha import MultiheadAttention
from models.dextra_unit import DExTraUnit

def _get_activation_fn(activation):
    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.dextra = DExTraUnit(in_features=d_model, in_proj_features=d_model, out_features=d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.gate1 = nn.Linear(d_model*2, d_model)
        self.gate1_act = F.gelu
        self.gate1_l2 = nn.Linear(d_model, d_model)

        self.gate2 = nn.Linear(d_model*2, d_model)
        self.gate2_act = F.gelu
        self.gate2_l2 = nn.Linear(d_model, d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # src_ = self.dextra(src)
        src_ = src
        src2 = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        g2 = torch.sigmoid(self.gate1_l2(self.gate1_act(self.gate1(torch.cat((src, src2), dim=2)))))
        src2 = src2 * g2
        src = src + self.dropout1(src2)*0.5
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # g2 = torch.sigmoid(self.gate2(torch.cat((src, src2), dim=2)))
        g2 = torch.sigmoid(self.gate2_l2(self.gate2_act(self.gate2(torch.cat((src, src2), dim=2)))))
        src2 = src2 * g2
        src = src + self.dropout2(src2)*0.5
        src = self.norm2(src)
        return src

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        self.config = config
        s2v_dim = config['s2v_dim']

        # input
        in_dr = config['sentence_encoder']['input_drop']
        self.in_dr = nn.Dropout(in_dr)

        # gated transformer
        word_edim = config['word_edim']
        num_head = config['sentence_encoder']['transformer']['num_heads']
        num_layers = config['sentence_encoder']['transformer']['num_layers']
        dim_feedforward = config['sentence_encoder']['transformer']['ffn_dim']
        tr_drop = config['sentence_encoder']['transformer']['residual_dropout']

        encoder_layer = TransformerEncoderLayer(d_model=word_edim, nhead=num_head,
                                                dim_feedforward=dim_feedforward, dropout=tr_drop)
        self.gtr = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # mha pool
        mha_nhead = config['sentence_encoder']['pooling']['mha']['num_heads']
        mha_dim = config['sentence_encoder']['pooling']['mha']['inner_dim']
        mha_drop = config['sentence_encoder']['pooling']['mha']['attention_dropout']
        out_dim_mult = int(mha_dim/word_edim)

        self.mha_pool = MultiheadAttention(word_edim, mha_nhead, dropout=mha_drop, out_dim_mult=out_dim_mult)

        # nli classifier
        class_dropout = config['classifier_network']['in_dropout']
        class_hdim = config['classifier_network']['hidden_dim']
        class_hact = config['classifier_network']['hidden_activation']
        num_classes = config['classifier_network']['num_classes']

        # self.fc1 = nn.Linear(4*s2v_dim, class_hdim)
        self.fc1_dr = nn.Dropout(class_dropout)
        self.fc1 = nn.Linear(4*mha_dim, class_hdim)
        self.act1 = _get_activation_fn(class_hact)
        self.fc2 = nn.Linear(class_hdim, num_classes)

    def _emb_sent(self, sent, sent_mask):
        sent = sent.permute((1, 0, 2))
        sent = self.gtr(sent, src_key_padding_mask=sent_mask)
        sent, _ = self.mha_pool(sent, sent, sent, key_padding_mask=sent_mask)
        sent = sent.permute((1, 0, 2))
        # print(sent_mask.shape)
        # print(sent.shape)
        # s2v = torch.max(sent + sent_mask*1e-5, axis=1)
        s2v, _ = torch.max(sent, axis=1)
        # s2v = torch.mean(sent, axis=1)
        return s2v

    def forward(self, sent1, sent2, sent1_mask=None, sent2_mask=None):
        sent1 = self.in_dr(sent1)
        sent2 = self.in_dr(sent2)

        s2v_sent1 = self._emb_sent(sent1, sent1_mask)
        s2v_sent2 = self._emb_sent(sent2, sent2_mask)

        x_class_in = torch.cat((s2v_sent1, s2v_sent2, torch.abs(s2v_sent1-s2v_sent2), s2v_sent1*s2v_sent2), dim=1)
        x_class_in = self.fc1_dr(x_class_in)
        x_class = self.fc1(x_class_in)
        x_class = self.act1(x_class)
        x_class = self.fc2(x_class)

        return s2v_sent1, s2v_sent2, x_class
