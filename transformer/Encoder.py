import torch
import torch.nn as nn
from neural_net import MultiHeadAttention, PositionWiseFFN, AddNorm


class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    #num hidden is to project the X to q k v space
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size,
                                                value_size, num_hiddens,
                                                num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class encoderr(nn.Module):
    """Transformer encoder."""
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(encoderr, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.pos_encoding = nn.Parameter(torch.rand([num_hiddens]))
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X += self.pos_encoding
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # self.attention_weights[
            #     i] = blk.attention.attention.attention_weights
        return X

if __name__ == '__main__':
    encoder = encoderr(24,24,24,24,[100,24],24,48,8,10,0.5)
    X = torch.rand([1,100,24])
    print(encoder(X,None).shape)