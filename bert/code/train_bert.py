import torch
from torch import nn
from d2l import torch as d2l
import data_process
import bert_model

def gpu(i):
    return torch.device(f'cuda:{i}')

def try_all_gpus():
#返回torch.device()列表
    return [gpu(i) for i in range(num_gpus())]
def num_gpus():
    return torch.cuda.device_count()

def get_gpu_else_cpu():
    cnt = num_gpus()
    if cnt >= 1:
        print(f'using {cnt} gpus')
        return try_all_gpus()
    else:
        print("can't access gpus")
        return [torch.device('cpu')]

#计算一个批量的损失之和
def _get_batch_loss_bert(net , loss , vocab_size , tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_Y):
    _,  mlm_Y_hat , nsp_Y_hat = net(tokens_X, segments_X,
                                    valid_lens_x, pred_positions_X)
    mlm_l = loss(mlm_Y_hat.reshape(-1 , vocab_size) , mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1 , 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    nsp_l = loss(nsp_Y_hat.reshape(-1,  2) , nsp_Y)
    l = mlm_l + nsp_l
    return mlm_l , nsp_l , l




def train_bert(train_iter, net , loss , vocab_size, devices, num_steps):
    devices = devices[1:] 
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters() , lr=0.01)
    step , timer = 0 , d2l.Timer()
    # animator = d2l.Animator(xlabel='step', ylabel='loss',
    #                         xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # metric = d2l.Accumulator(4)
    while step < num_steps:
        total_l = 0
        num_batches = 0
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, \
             mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l , nsp_l , l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y
            )
            total_l += l
            num_batches += 1
            l.backward()
            trainer.step()
            # metric.add(mlm_l, nsp_l, tokens_X.shape[0] , 1)
            timer.stop()
        avarage_l = total_l / num_batches
        step += 1
        if step % 50 == 0:
            torch.save(net, f'bert{step}.pth')
        print(f'total loss:{total_l} \n num_batches:{num_batches}')
        print(f'epoch {step} loss:{avarage_l}')



            # animator.add(step + 1 , (metric[0] / metric[3], metric[1] / metric[3]))
            # print(step)
            # step += 1

    # print(f'MLM loss {metric[0] / metric[3]:.3f}, '
    #       f'NSP loss {metric[1] / metric[3]:.3f}')
    # print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
    #       f'{str(devices)}')


devices = get_gpu_else_cpu()
loss = nn.CrossEntropyLoss()
if __name__ == '__main__':
    train_iter, vocab = data_process.load_data_wiki(batch_size=512, max_len=64)
    net = bert_model.BERTModel(vocab_size=len(vocab), num_hiddens=128 * 2, norm_shape=128*2,
                               ffn_num_input=128*2, ffn_num_hiddens=2 *  2 * 128, num_heads=4*2,
                               num_layers=4*2, dropout=0.2, max_len=64, key_size=128 *2, query_size=128*2,
                               value_size=128*2, hid_in_features=128*2,
                               mlm_in_features=128  * 2, nsp_in_features=128 * 2)

    train_bert(train_iter , net , loss , len(vocab) , devices , 151)

    net1 = torch.load('bert50.pth')
    net2 = torch.load('bert100.pth')
    net3 = torch.load('bert150.pth')

    tokens_a, token_b = ['what', 'a', 'nice', 'day'], ['i', 'am', 'very', '<mask>']
    tokens, segments = data_process.get_tokens_and_segments(tokens_a, token_b)
    # print(tokens[9])
    tokens_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    pred_positions = torch.tensor([9],device=devices[0]).unsqueeze(0)
    _,  d1, _ = net1(tokens_ids, segments, valid_len, pred_positions)
    _,  d2, _ = net2(tokens_ids, segments, valid_len, pred_positions)
    _,  d3, _ = net3(tokens_ids, segments, valid_len, pred_positions)
    
    print(d1[:, : 10])
    print(d2[:, : 10])
    print(d3[:,  :10])

    idx1 = torch.argmax(d1[0][0], dim = 0).item()
    print('bert50 predicition: what a nice day, i am very '+ vocab.to_token(idx1))

    idx2 = torch.argmax(d2[0][0], dim=0).item()
    print('bert100 predicition: what a nice day, i am very ' + vocab.to_token(idx2))

    idx3 = torch.argmax(d3[0][0], dim=0).item()
    print('bert150 predicition: what a nice day, i am very ' + vocab.to_token(idx3))
