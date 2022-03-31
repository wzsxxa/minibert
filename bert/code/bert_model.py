import torch
from torch import nn
from d2l import torch as d2l
import math


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads,dropout,
            use_bias
        )
        self.addnorm1 = AddNorm(norm_shape , dropout)
        self.ffn  = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape , dropout)

    def forward(self, X , valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape= norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size , num_hiddens, bias = bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        valid_lens  = torch.repeat_interleave(valid_lens, repeats=self.num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X , num_heads):
    X = X.reshape(X.shape[0] , X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2 , 1 , 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens = None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2 )) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores , valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def masked_softmax(scores, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(scores, dim = -1)
    else:
        shape  = scores.shape
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        scores = sequence_mask(scores.reshape(-1, shape[-1]) , valid_lens, value = -1e6)
        return nn.functional.softmax(scores.reshape(shape), dim=-1)


def sequence_mask(X, valid_lens, value):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, : ] < valid_lens[: , None]
    X[~mask] = value
    return X



class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self , vocab_size,  num_hiddens , norm_shape , ffn_num_input, ffn_num_hiddens ,
                 num_heads , num_layers , dropout , max_len = 1000 , key_size=768 , query_size = 768 , value_size = 768
                 , **kwargs):
        super(BERTEncoder , self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size , num_hiddens)
        self.segment_embedding = nn.Embedding(2 , num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(key_size , query_size, value_size , num_hiddens,
                                                          norm_shape ,ffn_num_input , ffn_num_hiddens, num_heads , dropout
                                                          , True))
        self.pos_embedding = nn.Parameter(torch.randn(1 , max_len , num_hiddens))

    def forward(self, tokens , segments , valid_lens):
        #下列代码中，X的形状保持不变：（批量大小，最大序列长度，  num_hiddens）
        #这里tokens的形状为（批量大小， 最大序列长度），然后内容为每一个单词对应的序号，把tokens喂入token_embedding后，
        #embedding将tokens中的每一个数字变成对应的词向量，所以最终得到的形状是（批量大小，最大序列长度，  num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)

        #这里有点疑惑，X的shape[1]即第二维是最大序列长度,pos_embeding.data的第二维也是max_len,这里切片为什么要这么写，
        #难道这两个最大长度不一样？ 可以肯定的是位置编码的长度肯定是BERTEncoder接收的序列长度。
        X = X + self.pos_embedding.data[:, :X.shape[1] ,  :] #
        for blk in self.blks:
            X = blk(X, valid_lens)

        return X

class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768 , **kwargs):
        super(MaskLM , self).__init__(**kwargs)
        #mlp的输入是一个被遮住的词向量，，输出是一个词库大小的一维向量，表示预测被遮住词在词库上的分布
        self.mlp = nn.Sequential(nn.Linear(num_inputs , num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens , vocab_size))

    #X的形状：（批量大小，最大序列长度，  num_hiddens）
    #pred_positions形状：（批量大小， 一个序列预测的单词数）
    def forward(self , X , pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1) #展为一维
        batch_size = X.shape[0]
        batch_idx = torch.arange(0 , batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X  = X[batch_idx , pred_positions]
        masked_X = masked_X.reshape((batch_size , num_pred_positions , -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat



class NextSentencePred(nn.Module):
    def __init__(self , num_inputs , **kwargs):
        super(NextSentencePred , self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs , 2)

    def forward(self , X):
        return self.output(X)



class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=128*2, query_size=128*2, value_size=128*2,
                 hid_in_features=128*2, mlm_in_features=128*2,
                 nsp_in_features=128*2):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                dropout, max_len=max_len, key_size=key_size,
                query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens , segments, valid_lens = None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None

        nsp_Y_hat = self.nsp(self.hidden(encoded_X[: , 0 , :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


if __name__ == '__main__':
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)
    print(encoded_X.shape)
    pred_positions = torch.tensor([[1,5 ,2 ] , [6 , 1 , 5]])
    mlm = MaskLM(vocab_size , 1000)
    mlm_Y_hat = mlm(encoded_X , pred_positions)
    print(mlm_Y_hat.shape)
    encoded_X = torch.flatten(encoded_X, start_dim=1)
    print(encoded_X.shape)
    # NSP的输⼊形状:(batchsize，num_hiddens)
    nsp = NextSentencePred(encoded_X.shape[-1])
    nsp_Y_hat = nsp(encoded_X)
    print(nsp_Y_hat.shape)
