import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modified_mha import MultiheadAttention
from models.dextra_unit import DExTraUnit
from torch.nn import Parameter

class NAC(nn.Module):
    # based on https://github.com/arthurdouillard/nalu.pytorch/blob/master/nalu.py
    def __init__(self, in_dim, out_dim, init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._W_hat = nn.Parameter(torch.empty(in_dim, out_dim))
        self._M_hat = nn.Parameter(torch.empty(in_dim, out_dim))

        self.register_parameter('W_hat', self._W_hat)
        self.register_parameter('M_hat', self._M_hat)

        for param in self.parameters():
            init_fun(param)

    def forward(self, x):
        W = F.tanh(self._W_hat) * F.sigmoid(self._M_hat)
        return x.matmul(W)

def round_act(x):
    return x - torch.sin(10*x*3.14)/(10*3.14)

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

class TransformerEncoderLayer_2(nn.Module):
    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_2, self).__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_2, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_ = self.norm1(src)
        src2 = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        src_ = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)

        return src

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

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, out_dim_mult=1.0)
        self.linear_mha = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.gate1 = nn.Linear(d_model*2, d_model)
        self.gate2 = nn.Linear(d_model*2, d_model)

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
        src2 = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear_mha(F.gelu(src2))
        g2 = self.gate1(torch.cat((src_, src2), dim=2))
        g2_shape = g2.shape
        g2 = g2.reshape((g2_shape[0], g2_shape[1], 128, 8))
        g2 = torch.softmax(g2, dim=-1)
        g2 = g2.reshape((g2_shape[0], g2_shape[1], g2_shape[2]))
        src2 = src2 * g2
        src = src + self.dropout1(src2)

        src_ = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        g2 = self.gate2(torch.cat((src_, src2), dim=2))
        g2_shape = g2.shape
        g2 = g2.reshape((g2_shape[0], g2_shape[1], 128, 8))
        g2 = torch.softmax(g2, dim=-1)
        g2 = g2.reshape((g2_shape[0], g2_shape[1], g2_shape[2]))
        src2 = src2 * g2
        src = src + self.dropout2(src2)

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
        out_dim_mult = mha_dim/word_edim

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

        self.zs_fc1_dr = nn.Dropout(class_dropout)
        self.zs_fc1 = nn.Linear(4*mha_dim, class_hdim)
        self.zs_act1 = _get_activation_fn(class_hact)
        self.zs_fc2 = nn.Linear(class_hdim, 2)

        self.qq_fc1_dr = nn.Dropout(class_dropout)
        self.qq_fc1 = nn.Linear(4*mha_dim, class_hdim)
        self.qq_act1 = _get_activation_fn(class_hact)
        self.qq_fc2 = nn.Linear(class_hdim, 2)

        self.blimp_fc1 = nn.Linear(mha_dim, 128)
        self.blimp_act1 = _get_activation_fn(class_hact)
        self.blimp_fc2 = nn.Linear(128, 2)

        self.paws_fc1_dr = nn.Dropout(class_dropout)
        self.paws_fc1 = nn.Linear(4*mha_dim, class_hdim)
        self.paws_act1 = _get_activation_fn(class_hact)
        self.paws_fc2 = nn.Linear(class_hdim, 2)

    def _emb_sent(self, sent, sent_mask):
        sent = self.in_dr(sent)

        sent = sent.permute((1, 0, 2))

        sent = self.gtr(sent, src_key_padding_mask=sent_mask)
        sent_ = self.norm2(sent)

        sent, _ = self.mha_pool(sent_, sent_, sent_, key_padding_mask=sent_mask)

        sent = sent.permute((1, 0, 2))

        sent_mask_exp = torch.cat([sent_mask.unsqueeze(2)]*sent.shape[2], dim=2).type(torch.cuda.FloatTensor)
        # s2v, _ = torch.max(sent + sent_mask_exp*-1e3, axis=1)
        s2v = torch.sum(sent*(1-sent_mask_exp), axis=1) / torch.sum((1-sent_mask_exp), axis=1)

        # s2v = s2v + torch.mean(s2v)*torch.randn_like(s2v)*0.1

        return s2v

    def forward(self, sent1, sent2, sent1_mask=None, sent2_mask=None):
        s2v_sent1 = self._emb_sent(sent1, sent1_mask)
        s2v_sent2 = self._emb_sent(sent2, sent2_mask)

        x_class_in = torch.cat((s2v_sent1, s2v_sent2, torch.abs(s2v_sent1-s2v_sent2), s2v_sent1*s2v_sent2), dim=1)
        x_class_in = self.fc1_dr(x_class_in)
        x_class = self.fc1(x_class_in)
        x_class = self.act1(x_class)
        nli_class = self.fc2(x_class)

        zs_class_in = torch.cat((s2v_sent1, s2v_sent2, torch.abs(s2v_sent1-s2v_sent2), s2v_sent1*s2v_sent2), dim=1)
        zs_class_in = self.zs_fc1_dr(zs_class_in)
        zs_class = self.zs_fc1(zs_class_in)
        zs_class = self.zs_act1(zs_class)
        zs_class = self.zs_fc2(zs_class)

        qq_class_in = torch.cat((s2v_sent1, s2v_sent2, torch.abs(s2v_sent1-s2v_sent2), s2v_sent1*s2v_sent2), dim=1)
        qq_class_in = self.qq_fc1_dr(qq_class_in)
        qq_class = self.qq_fc1(qq_class_in)
        qq_class = self.qq_act1(qq_class)
        qq_class = self.qq_fc2(qq_class)

        blimp_class_in = s2v_sent1
        blimp_class = self.blimp_fc1(blimp_class_in)
        blimp_class = self.blimp_act1(blimp_class)
        blimp_class = self.blimp_fc2(blimp_class)

        paws_class_in = torch.cat((s2v_sent1, s2v_sent2, torch.abs(s2v_sent1-s2v_sent2), s2v_sent1*s2v_sent2), dim=1)
        paws_class_in = self.paws_fc1_dr(paws_class_in)
        paws_class = self.paws_fc1(paws_class_in)
        paws_class = self.paws_act1(paws_class)
        paws_class = self.paws_fc2(paws_class)

        return {
            's2v_sent1': s2v_sent1,
            's2v_sent2': s2v_sent2,
            'nli_pred': nli_class,
            'zs_pred': zs_class,
            'qq_pred': qq_class,
            'blimp_pred': blimp_class,
            'paws_pred': paws_class
        }
