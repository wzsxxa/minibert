import collections
import os
import random
import torch
from d2l import torch as d2l

DATA_HUB = dict()
DATA_HUB['wikitext-2'] = ('https://s3.amazonaws.com/research.metamind.io/wikitext/'
'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')


#获取输入序列的词元及其片段索引
#0和1分别标记片段A和B
def get_tokens_and_segments(tokens_a , tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens , segments

#返回的paragraphs是一个段落列表 ， 每一个段落又是一个句子列表。
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir , 'wiki.train.tokens')
    with open(file_name  , 'r' , encoding='utf-8' ) as f:
        lines  = f.readlines()
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def _get_next_sentence(sentence , next_sentence , paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        #书上面说这里的paragraphs是三重列表，因为每个句子还是一个单词列表。
        # 所以这里的paragrahs应该和上面的paragraphs不一样
        #上面的paragraphs最小元素就是句子了，是二重列表
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False

    return sentence , next_sentence ,  is_next



#下⾯的函数通过调⽤_get_next_sentence函数从输⼊paragraph⽣成⽤于下⼀句预测的训练样本。这
# ⾥paragraph是句⼦列表，其中每个句⼦都是词元列表。⾃变量max_len指定预训练期间的BERT输⼊序列
# 的最⼤⻓度。
def _get_nsp_data_from_paragraph(paragraph , paragraphs , vocab , max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a , tokens_b , is_next  = _get_next_sentence(
            paragraph[i] , paragraph[i + 1] , paragraphs
        )
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments  = get_tokens_and_segments(tokens_a , tokens_b)
        nsp_data_from_paragraph.append((tokens , segments , is_next))
    return nsp_data_from_paragraph

#获取mlm任务的数据，tokens时表示BERT输入序列的词元的列表
def _replace_mlm_tokens(tokens , candidate_pred_position, num_mlm_preds,
                        vocab):
    mlm_input_tokens = [token for token in tokens]
    pred_position_and_labels = []
    random.shuffle(candidate_pred_position)
    for mlm_pred_position in candidate_pred_position:
        if len(pred_position_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random()  < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_position_and_labels.append((mlm_pred_position , tokens[mlm_pred_position]))
    return mlm_input_tokens  , pred_position_and_labels


# 通过调⽤前述的_replace_mlm_tokens函数，以下函数将BERT输⼊序列（tokens）作为输⼊，并返回输
# ⼊词元的索引、发⽣预测的词元索引以及这些预测的标签索引。
def _get_mlm_data_from_tokens(tokens , vocab):
    candidate_pred_positions = []
    for i , token in enumerate(tokens):
        if token in ['<cls>'  , '<sep>']:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds = max(1,  round(len(tokens) * 0.15))
    mlm_input_tokens , pred_positions_and_labels = _replace_mlm_tokens(
        tokens , candidate_pred_positions , num_mlm_preds , vocab
    )
    pred_positions_and_labels = sorted(pred_positions_and_labels ,
                                       key = lambda x : x[0])
    pred_positions  = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens] , pred_positions , vocab[mlm_pred_labels]

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0] , list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# Vocab类实现的功能：实现词元和索引的相互转化
class Vocab:
    def __init__(self , sentences , min_freq , reserved_tokens):
        counter = count_corpus(sentences)
        #key = lambda x  : x[1]的含义， 其中x表示列表元素，这里是一个键值对。
        # x[1]表示按照键值对中的值进行排序
        self.token_freqs  =sorted(counter.items() , key= lambda  x : x[1] ,
                                  reverse= True)
        self.unk , uniq_tokens = 0 , ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token , freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token , self.token_to_idx = [] , dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

#定义__len__后  , 再对Vocab实例使用len()函数时 会调用 __len__()
    def __len__(self):
        return len(self.idx_to_token)

#定义__getitem__后， 再对Vocab实例使用 [] 时 会调用 __getitem__()
    def __getitem__(self, tokens):
        return [self.token_to_idx.get(token , self.unk) for token in tokens]


#给定一个索引， 将其转化为token
    def to_token(self , idx):
        return self.idx_to_token[idx]


# 定 义 辅 助 函
# 数_pad_bert_inputs来将特殊的“<mask>”词元附加到输⼊。它的参数examples包含来⾃两个预训练
# 任务的辅助函数_get_nsp_data_from_paragraph和_get_mlm_data_from_tokens的输出。

def _pad_bert_inputs(examples , max_len , vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids , all_segments , valid_lens =[] , [] , []
    all_pred_positions , all_mlm_weights , all_mlm_labels =[] ,[] ,[]
    nsp_labels = []
    i = 1
    for (token_ids , pred_positions , mlm_pred_labels_ids, segments ,
         is_next) in examples:
        # print(i)
        # i += 1
        all_token_ids.append(torch.tensor(token_ids + vocab[['<pad>']] *
                                          (max_len - len(token_ids)) , dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)) ,
                                         dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids) , dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0]*(max_num_mlm_preds -
                                        len(pred_positions)) ,dtype=torch.long))
        #填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_labels_ids) +
                    [0.0] * (max_num_mlm_preds - len(mlm_pred_labels_ids)) ,
                                            dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_labels_ids + [0] *
                                           (max_num_mlm_preds - len(mlm_pred_labels_ids)),
                                           dtype=torch.long) )
        nsp_labels.append(torch.tensor(is_next , dtype = torch.long))
    return (all_token_ids , all_segments , valid_lens , all_pred_positions,
            all_mlm_weights , all_mlm_labels , nsp_labels)

def tokenize(lines , token = 'word'):
    return [line.split() if token == 'word' else list(line) for line in lines]


class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self , paragraphs , max_len):
        # 输⼊paragraphs[i]是代表段落的句⼦字符串列表；
        # ⽽输出paragraphs[i]是代表段落的句⼦列表，其中每个句⼦都是词元列表
        paragraphs = [tokenize(paragraph)  for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = Vocab(sentences , min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'
        ])
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len
            ))
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]

        #填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(examples, max_len,
                                                                  self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    num_workers = 0
    data_dir = '../data/wikitext-2'
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                            shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab


if __name__ == '__main__':
    batch_size, max_len = 512, 300
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    # print(vocab['<pad>'])
    # print("xxxx")
    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
         mlm_Y, nsp_y) in train_iter:
        # print(tokens_X[0], segments_X[0], valid_lens_x[0],
        #       pred_positions_X[0], mlm_weights_X[0], mlm_Y[0],
        #       nsp_y[0])
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
              pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
              nsp_y.shape)
        break

    print(vocab[['<pad>']])

    # data_dir = '..\data\wikitext-2'
    # paragraphs = _read_wiki(data_dir)
    # for i in range(3):
    #     print(paragraphs[i])