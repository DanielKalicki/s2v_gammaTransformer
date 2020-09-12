import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modified_mha import MultiheadAttention
from models.dextra_unit import DExTraUnit
from torch.nn import Parameter


class DenseLayer(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, output_dim=4096,
                 drop=0.0, hidden_act='relu', conv=False, cat_dim=1):
        super(DenseLayer, self).__init__()
        self.hidden_act = hidden_act
        self.conv = conv
        self.cat_dim = cat_dim

        if not self.conv:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(input_dim+hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(input_dim+2*hidden_dim, output_dim)
        else:
            self.conv_1 = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
            self.conv_2 = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 3, padding=1)
            self.conv_3 = nn.Conv1d(input_dim+2*hidden_dim, output_dim, 3, padding=1)

        if self.hidden_act == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        self.drop1 = nn.Dropout(p=drop)
        self.drop2 = nn.Dropout(p=drop)

    def forward(self, x):
        # dense layer 1
        x1_in = x
        if not self.conv:
            x1 = self.fc1(x1_in)
        else:
            x1 = self.conv_1(x1_in.permute(0,2,1)).permute(0,2,1)
        x1 = self.act1(x1)
        x1 = self.drop1(x1)

        # dense layer 2
        x2_in = torch.cat((x, x1), dim=self.cat_dim)
        if not self.conv:
            x2 = self.fc2(x2_in)
        else:
            x2 = self.conv_2(x2_in.permute(0,2,1)).permute(0,2,1)
        x2 = self.act2(x2)
        x2 = self.drop2(x2)

        # dense layer 3
        x3_in = torch.cat((x, x1, x2), dim=self.cat_dim)
        if not self.conv:
            y = self.fc_out(x3_in)
        else:
            y = self.conv_3(x3_in.permute(0,2,1)).permute(0,2,1)
        return y


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
        # self.linear2 = DenseLayer(dim_feedforward, dim_feedforward, d_model, cat_dim=2)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.linear1_0 = nn.Linear(d_model, dim_feedforward)
        # self.linear1_1 = nn.Linear(d_model, dim_feedforward)
        # self.linear1_2 = nn.Linear(d_model, dim_feedforward)
        # self.linear1_3 = nn.Linear(d_model, dim_feedforward)
        # self.dropout_0 = nn.Dropout(0.1)
        # self.dropout_1 = nn.Dropout(0.1)
        # self.dropout_2 = nn.Dropout(0.1)
        # self.dropout_3 = nn.Dropout(0.1)

        # self.linear1_mha = nn.Linear(d_model//4, d_model)
        # self.linear1_mha = DenseLayer(d_model, dim_feedforward, dim_feedforward, cat_dim=2)
        # self.activation_mha = _get_activation_fn(activation)
        # self.dropout_mha = nn.Dropout(dropout)
        # self.linear2_mha = nn.Linear(d_model, d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, out_dim_mult=0.5)
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, out_dim_mult=0.25)
        # self.linear_mha = nn.Linear(d_model//2, d_model)
        # self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout, out_dim_mult=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm1_ = nn.LayerNorm(d_model*2)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm2_ = nn.LayerNorm(d_model*2)

        self.gate1 = nn.Linear(d_model*2, d_model)
        # self.gate1 = MultiheadAttention(d_model*2, nhead, dropout=dropout, out_dim_mult=0.5)
        # self.gate1_act = F.gelu
        # self.gate1_l2 = nn.Linear(d_model, d_model)

        # self.gate1_rnn = nn.GRU(d_model, d_model//2, bidirectional=True)
        # self.gate2_rnn = nn.GRU(d_model, d_model//2, bidirectional=True)

        self.gate2 = nn.Linear(d_model*2, d_model)
        # self.gate2 = MultiheadAttention(d_model*2, nhead, dropout=dropout, out_dim_mult=0.5)
        # self.gate2_act = F.gelu
        # self.gate2_l2 = nn.Linear(d_model, d_model)

        # self.src_gate = nn.Linear(d_model, d_model)

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
        src_ = self.norm1(src)
        # src_ = src
        src2 = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # src2 = self.self_attn2(src2, src2, src2, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        # src2 = self.self_attn2(src2, src2, src2, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        # src2 = self.linear_mha(F.gelu(src2))
        # g2 = torch.sigmoid(self.gate1_l2(self.gate1_act(self.gate1(torch.cat((src_, src2), dim=2)))))
        g2 = torch.sigmoid(self.gate1(torch.cat((src_, src2), dim=2)))
        # src_cat = torch.cat((src_, src2), dim=2)
        # g2 = torch.sigmoid(self.gate1_l2(self.gate1_act(self.gate1(src_cat,src_cat,src_cat, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0])))
        # g2 = torch.tanh(torch.abs(self.gate1(self.gate1_rnn(src2)[0])))
        # g2 = torch.tanh(torch.abs(src2))
        # src2 = self.gate1_l2(self.gate1_act(self.gate1(torch.cat((src_, src2), dim=2))))
        src2 = src2 * g2
        src = src + self.dropout1(src2)*0.25
        # src = self.norm1(src)

        src_ = self.norm2(src)
        # src_ = self.norm2(torch.cat((src, src2), dim=2))
        # src_ = src
        # src_ = (self.dropout_0(self.linear1_0(src_)) + self.dropout_1(self.linear1_1(src_)) + self.dropout_2(self.linear1_2(src_)) + self.dropout_3(self.linear1_3(src_))) / 4
        # src2 = self.linear2(self.dropout(self.activation(src_)))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        # src2 = self.conv1(src.permute(1,2,0)).permute(2,0,1)
        # src2 = self.linear2(self.dropout(self.activation(src2)))
        # g2 = torch.sigmoid(self.gate2(torch.cat((src, src2), dim=2)))
        # g2 = torch.sigmoid(self.gate2_l2(self.gate2_act(self.gate2(torch.cat((src_, src2), dim=2)))))
        # g2 = torch.sigmoid(self.gate2(torch.cat((src_, src2), dim=2)))
        g2 = torch.sigmoid(self.gate2(torch.cat((src_, src2), dim=2)))
        # src_cat = torch.cat((src_, src2), dim=2)
        # g2 = torch.sigmoid(self.gate2_l2(self.gate2_act(self.gate2(src_cat,src_cat,src_cat, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0])))
        # g2 = torch.tanh(torch.abs(self.gate2(src2)))
        # g2 = torch.tanh(torch.abs(self.gate2(self.gate2_rnn(src2)[0])))
        # g2 = torch.tanh(torch.abs(src2))
        # src2 = self.gate2(src_)
        # g2 = self.gate2_l2(self.gate2_act(self.gate2(torch.cat((src_, src2), dim=2))))
        src2 = src2 * g2
        src = src + self.dropout2(src2)*0.25
        # src = self.norm2(src)

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

        self.norm2 = nn.LayerNorm(word_edim)

        encoder_layer = TransformerEncoderLayer(d_model=word_edim, nhead=num_head,
                                                dim_feedforward=dim_feedforward, dropout=tr_drop, activation="gelu")
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

        self.fc1_dr = nn.Dropout(class_dropout)
        self.fc1 = nn.Linear(4*mha_dim, class_hdim)
        self.act1 = _get_activation_fn(class_hact)
        self.fc2 = nn.Linear(class_hdim, num_classes)

    def _emb_sent(self, sent, sent_mask):
        sent = self.in_dr(sent)

        sent = sent.permute((1, 0, 2))
        # sent = self.norm1(sent)
        # sent = F.gelu(sent)

        sent = self.gtr(sent, src_key_padding_mask=sent_mask)
        # sent_mha_gate, _ = self.mha_pool_1h(sent, sent, sent, key_padding_mask=sent_mask)
        # sent_g = torch.sigmoid(self.sent_gate_l2(F.gelu(self.sent_gate_l1(sent_mha_gate))))
        sent = self.norm2(sent)
        sent_ = sent

        # k = F.gelu(self.q_fc(sent))
        
        # sent = torch.cat([sent, sent_g], dim=2)

        # sent = torch.cat([sent, sent2, sent3], dim=2)
        sent, _ = self.mha_pool(sent, sent, sent, key_padding_mask=sent_mask)

        # sent1h, _ = self.mha_pool_1h(sent, sent, sent, key_padding_mask=sent_mask)
        # sent4h, _ = self.mha_pool_4h(sent, sent, sent, key_padding_mask=sent_mask)
        # sent16h, _ = self.mha_pool_16h(sent, sent, sent, key_padding_mask=sent_mask)
        # sent64h, _ = self.mha_pool_64h(sent, sent, sent, key_padding_mask=sent_mask)
        # g1h = F.sigmoid(self.mha_pool_1h_gate(sent))
        # g4h = F.sigmoid(self.mha_pool_4h_gate(sent))
        # g16h = F.sigmoid(self.mha_pool_16h_gate(sent))
        # g64h = F.sigmoid(self.mha_pool_64h_gate(sent))
        # sent = sent + g1h*sent1h + g4h*sent4h + g16h*sent16h + g64h*sent64h

        sent = sent.permute((1, 0, 2))

        sent_mask_exp = torch.cat([sent_mask.unsqueeze(2)]*sent.shape[2], dim=2).type(torch.cuda.FloatTensor)
        # s2v, _ = torch.max(sent + sent_mask_exp*-1e3, axis=1)

        s2v = torch.sum(sent*(1-sent_mask_exp), axis=1) / torch.sum((1-sent_mask_exp), axis=1)

        return s2v

    def forward(self, sent1, sent2, sent1_mask=None, sent2_mask=None, test_words1=None, test_words2=None):
        s2v_sent1 = self._emb_sent(sent1, sent1_mask)
        s2v_sent2 = self._emb_sent(sent2, sent2_mask)

        x_class_in = torch.cat((s2v_sent1, s2v_sent2, torch.abs(s2v_sent1-s2v_sent2), s2v_sent1*s2v_sent2), dim=1)
        x_class_in = self.fc1_dr(x_class_in)
        x_class = self.fc1(x_class_in)
        x_class = self.act1(x_class)
        x_class = self.fc2(x_class)

        return s2v_sent1, s2v_sent2, x_class
